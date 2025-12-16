//! Binary format for efficient retina storage.
//!
//! The format is designed for fast random access and minimal storage overhead.
//!
//! ## Format Layout
//!
//! ```text
//! +------------------+
//! | Header (64 bytes)|
//! +------------------+
//! | Word Index Table |
//! | (variable)       |
//! +------------------+
//! | Fingerprint Data |
//! | (variable)       |
//! +------------------+
//! | String Pool      |
//! | (variable)       |
//! +------------------+
//! | Inverted Index   |  (optional, if HAS_INVERTED_INDEX flag set)
//! | (variable)       |
//! +------------------+
//! ```
//!
//! ### Header (64 bytes)
//! - Magic number (4 bytes): "PRET"
//! - Version (2 bytes)
//! - Flags (2 bytes): bit 0 = has inverted index
//! - Grid dimension (4 bytes)
//! - Number of words (4 bytes)
//! - Index table offset (8 bytes)
//! - Fingerprint data offset (8 bytes)
//! - String pool offset (8 bytes)
//! - Inverted index offset (8 bytes)
//! - Reserved (16 bytes)
//!
//! ### Word Index Table
//! - Array of (string_offset: u32, string_len: u16, fp_offset: u32, fp_len: u16)
//!
//! ### Fingerprint Data
//! - Compressed RoaringBitmap data for each word
//!
//! ### String Pool
//! - Concatenated word strings (UTF-8)

use crate::error::{ProteusError, Result};
use crate::fingerprint::{Sdr, WordFingerprint};
use crate::index::InvertedIndex;
use memmap2::{Mmap, MmapOptions};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Magic number for Proteus retina files.
const MAGIC: &[u8; 4] = b"PRET";

/// Current format version.
const VERSION: u16 = 3;

/// Header size in bytes.
const HEADER_SIZE: usize = 64;

/// Flag indicating the file contains a persisted inverted index (bincode, v2).
const FLAG_HAS_INVERTED_INDEX: u16 = 0x0001;

/// Flag indicating the file contains a memory-mappable inverted index (v3).
const FLAG_HAS_MMAP_INDEX: u16 = 0x0002;

/// Retina file header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetinaHeader {
    /// Grid dimension.
    pub dimension: u32,
    /// Number of words.
    pub num_words: u32,
    /// Offset to the index table.
    pub index_offset: u64,
    /// Offset to fingerprint data.
    pub fingerprint_offset: u64,
    /// Offset to string pool.
    pub string_pool_offset: u64,
    /// Offset to inverted index (0 if not present).
    pub inverted_index_offset: u64,
    /// Format version.
    pub version: u16,
    /// Flags.
    pub flags: u16,
}

impl RetinaHeader {
    /// Returns true if this header indicates a bincode inverted index is present (v2).
    pub fn has_inverted_index(&self) -> bool {
        self.flags & FLAG_HAS_INVERTED_INDEX != 0
    }

    /// Returns true if this header indicates a memory-mappable inverted index is present (v3).
    pub fn has_mmap_index(&self) -> bool {
        self.flags & FLAG_HAS_MMAP_INDEX != 0
    }
}

impl RetinaHeader {
    /// Creates a new header.
    pub fn new(dimension: u32, num_words: u32) -> Self {
        Self {
            dimension,
            num_words,
            index_offset: HEADER_SIZE as u64,
            fingerprint_offset: 0, // Set during write
            string_pool_offset: 0, // Set during write
            inverted_index_offset: 0, // Set during write if index present
            version: VERSION,
            flags: 0,
        }
    }

    /// Creates a new header with memory-mappable inverted index flag set.
    pub fn new_with_index(dimension: u32, num_words: u32) -> Self {
        Self {
            dimension,
            num_words,
            index_offset: HEADER_SIZE as u64,
            fingerprint_offset: 0,
            string_pool_offset: 0,
            inverted_index_offset: 0,
            version: VERSION,
            flags: FLAG_HAS_MMAP_INDEX,
        }
    }

    /// Writes the header to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; HEADER_SIZE];

        // Magic number
        bytes[0..4].copy_from_slice(MAGIC);

        // Version
        bytes[4..6].copy_from_slice(&self.version.to_le_bytes());

        // Flags
        bytes[6..8].copy_from_slice(&self.flags.to_le_bytes());

        // Dimension
        bytes[8..12].copy_from_slice(&self.dimension.to_le_bytes());

        // Number of words
        bytes[12..16].copy_from_slice(&self.num_words.to_le_bytes());

        // Index offset
        bytes[16..24].copy_from_slice(&self.index_offset.to_le_bytes());

        // Fingerprint offset
        bytes[24..32].copy_from_slice(&self.fingerprint_offset.to_le_bytes());

        // String pool offset
        bytes[32..40].copy_from_slice(&self.string_pool_offset.to_le_bytes());

        // Inverted index offset
        bytes[40..48].copy_from_slice(&self.inverted_index_offset.to_le_bytes());

        // Reserved (bytes 48-63)
        bytes
    }

    /// Reads a header from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_SIZE {
            return Err(ProteusError::InvalidRetinaFormat(
                "Header too short".to_string(),
            ));
        }

        // Check magic number
        if &bytes[0..4] != MAGIC {
            return Err(ProteusError::InvalidRetinaFormat(
                "Invalid magic number".to_string(),
            ));
        }

        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        let flags = u16::from_le_bytes([bytes[6], bytes[7]]);
        let dimension = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let num_words = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);

        let index_offset = u64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
        ]);
        let fingerprint_offset = u64::from_le_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
        ]);
        let string_pool_offset = u64::from_le_bytes([
            bytes[32], bytes[33], bytes[34], bytes[35], bytes[36], bytes[37], bytes[38], bytes[39],
        ]);
        // For version 1 files, inverted_index_offset will be 0 (from reserved bytes)
        let inverted_index_offset = u64::from_le_bytes([
            bytes[40], bytes[41], bytes[42], bytes[43], bytes[44], bytes[45], bytes[46], bytes[47],
        ]);

        Ok(Self {
            dimension,
            num_words,
            index_offset,
            fingerprint_offset,
            string_pool_offset,
            inverted_index_offset,
            version,
            flags,
        })
    }
}

/// Index entry for a word.
#[derive(Debug, Clone, Copy)]
struct IndexEntry {
    /// Offset into the string pool.
    string_offset: u32,
    /// Length of the word string.
    string_len: u16,
    /// Offset into fingerprint data.
    fp_offset: u32,
    /// Length of fingerprint data.
    fp_len: u16,
}

impl IndexEntry {
    const SIZE: usize = 12;

    fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[0..4].copy_from_slice(&self.string_offset.to_le_bytes());
        bytes[4..6].copy_from_slice(&self.string_len.to_le_bytes());
        bytes[6..10].copy_from_slice(&self.fp_offset.to_le_bytes());
        bytes[10..12].copy_from_slice(&self.fp_len.to_le_bytes());
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            string_offset: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            string_len: u16::from_le_bytes([bytes[4], bytes[5]]),
            fp_offset: u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]),
            fp_len: u16::from_le_bytes([bytes[10], bytes[11]]),
        }
    }
}

/// Binary format reader/writer for retina files.
pub struct RetinaFormat;

impl RetinaFormat {
    /// Writes fingerprints to a binary file (without inverted index).
    pub fn write<P: AsRef<Path>>(
        path: P,
        fingerprints: &HashMap<String, WordFingerprint>,
        dimension: u32,
    ) -> Result<()> {
        Self::write_internal(path, fingerprints, dimension, None)
    }

    /// Writes fingerprints with an inverted index to a binary file.
    pub fn write_with_index<P: AsRef<Path>>(
        path: P,
        fingerprints: &HashMap<String, WordFingerprint>,
        dimension: u32,
        inverted_index: &InvertedIndex,
    ) -> Result<()> {
        Self::write_internal(path, fingerprints, dimension, Some(inverted_index))
    }

    fn write_internal<P: AsRef<Path>>(
        path: P,
        fingerprints: &HashMap<String, WordFingerprint>,
        dimension: u32,
        inverted_index: Option<&InvertedIndex>,
    ) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let num_words = fingerprints.len() as u32;
        let mut header = if inverted_index.is_some() {
            RetinaHeader::new_with_index(dimension, num_words)
        } else {
            RetinaHeader::new(dimension, num_words)
        };

        // Calculate offsets
        let index_size = (num_words as usize) * IndexEntry::SIZE;
        header.fingerprint_offset = header.index_offset + index_size as u64;

        // Collect all data
        let mut entries: Vec<IndexEntry> = Vec::with_capacity(num_words as usize);
        let mut string_pool = Vec::new();
        let mut fp_data = Vec::new();

        // Sort words for consistent ordering
        let mut sorted_words: Vec<&String> = fingerprints.keys().collect();
        sorted_words.sort();

        for word in sorted_words {
            let wf = &fingerprints[word];

            // Serialize fingerprint
            let mut fp_bytes = Vec::new();
            wf.fingerprint.bitmap().serialize_into(&mut fp_bytes)
                .map_err(|e| ProteusError::Serialization(e.to_string()))?;
            let fp_offset = fp_data.len() as u32;
            let fp_len = fp_bytes.len() as u16;
            fp_data.extend(fp_bytes);

            // Add string to pool
            let string_offset = string_pool.len() as u32;
            let string_len = word.len() as u16;
            string_pool.extend(word.as_bytes());

            entries.push(IndexEntry {
                string_offset,
                string_len,
                fp_offset,
                fp_len,
            });
        }

        header.string_pool_offset = header.fingerprint_offset + fp_data.len() as u64;

        // Build mmap-friendly inverted index if requested
        let (index_offset_table, index_posting_data) = if let Some(idx) = inverted_index {
            header.inverted_index_offset = header.string_pool_offset + string_pool.len() as u64;
            Self::build_mmap_index(idx)?
        } else {
            (Vec::new(), Vec::new())
        };

        // Write header
        writer.write_all(&header.to_bytes())?;

        // Write index
        for entry in &entries {
            writer.write_all(&entry.to_bytes())?;
        }

        // Write fingerprint data
        writer.write_all(&fp_data)?;

        // Write string pool
        writer.write_all(&string_pool)?;

        // Write mmap-friendly inverted index if present
        if !index_offset_table.is_empty() {
            // Write offset table
            writer.write_all(&index_offset_table)?;
            // Write posting list data
            writer.write_all(&index_posting_data)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Builds mmap-friendly inverted index data.
    ///
    /// Returns (offset_table, posting_data) where:
    /// - offset_table: grid_size * 8 bytes, each entry is (offset: u32, len: u32)
    /// - posting_data: concatenated serialized RoaringBitmaps
    fn build_mmap_index(idx: &InvertedIndex) -> Result<(Vec<u8>, Vec<u8>)> {
        let grid_size = idx.grid_size() as usize;

        let mut offset_table = Vec::with_capacity(grid_size * 8);
        let mut posting_data = Vec::new();

        for bitmap in idx.posting_lists_iter() {
            if bitmap.is_empty() {
                // Empty posting list
                offset_table.extend_from_slice(&0u32.to_le_bytes());
                offset_table.extend_from_slice(&0u32.to_le_bytes());
            } else {
                let offset = posting_data.len() as u32;

                bitmap
                    .serialize_into(&mut posting_data)
                    .map_err(|e| ProteusError::Serialization(e.to_string()))?;

                let len = posting_data.len() as u32 - offset;
                offset_table.extend_from_slice(&offset.to_le_bytes());
                offset_table.extend_from_slice(&len.to_le_bytes());
            }
        }

        Ok((offset_table, posting_data))
    }

    /// Reads fingerprints from a binary file (without loading inverted index).
    pub fn read<P: AsRef<Path>>(path: P) -> Result<(RetinaHeader, HashMap<String, WordFingerprint>)> {
        let (header, fingerprints, _) = Self::read_with_index(path)?;
        Ok((header, fingerprints))
    }

    /// Reads fingerprints and inverted index from a binary file.
    ///
    /// Returns (header, fingerprints, optional inverted index).
    pub fn read_with_index<P: AsRef<Path>>(
        path: P,
    ) -> Result<(RetinaHeader, HashMap<String, WordFingerprint>, Option<InvertedIndex>)> {
        let mut file = File::open(path)?;

        // Read header
        let mut header_bytes = [0u8; HEADER_SIZE];
        file.read_exact(&mut header_bytes)?;
        let header = RetinaHeader::from_bytes(&header_bytes)?;

        // Read index
        file.seek(SeekFrom::Start(header.index_offset))?;
        let index_size = (header.num_words as usize) * IndexEntry::SIZE;
        let mut index_bytes = vec![0u8; index_size];
        file.read_exact(&mut index_bytes)?;

        let entries: Vec<IndexEntry> = index_bytes
            .chunks_exact(IndexEntry::SIZE)
            .map(IndexEntry::from_bytes)
            .collect();

        // Read fingerprint data
        file.seek(SeekFrom::Start(header.fingerprint_offset))?;
        let fp_size = (header.string_pool_offset - header.fingerprint_offset) as usize;
        let mut fp_data = vec![0u8; fp_size];
        file.read_exact(&mut fp_data)?;

        // Read string pool (calculate size based on whether inverted index is present)
        file.seek(SeekFrom::Start(header.string_pool_offset))?;
        let string_pool_size = if header.has_inverted_index() {
            (header.inverted_index_offset - header.string_pool_offset) as usize
        } else {
            // Read to end of file
            let file_size = file.seek(SeekFrom::End(0))?;
            file.seek(SeekFrom::Start(header.string_pool_offset))?;
            (file_size - header.string_pool_offset) as usize
        };
        let mut string_pool = vec![0u8; string_pool_size];
        file.read_exact(&mut string_pool)?;

        // Reconstruct fingerprints
        let grid_size = header.dimension * header.dimension;
        let mut fingerprints = HashMap::with_capacity(header.num_words as usize);

        for entry in entries {
            // Extract word
            let start = entry.string_offset as usize;
            let end = start + entry.string_len as usize;
            let word = String::from_utf8_lossy(&string_pool[start..end]).to_string();

            // Extract fingerprint
            let fp_start = entry.fp_offset as usize;
            let fp_end = fp_start + entry.fp_len as usize;
            let bitmap =
                crate::roaring::RoaringBitmap::deserialize_unchecked_from(&fp_data[fp_start..fp_end])
                    .map_err(|e| ProteusError::Serialization(e.to_string()))?;

            let sdr = Sdr::from_bitmap(bitmap, grid_size);
            let wf = WordFingerprint {
                word: word.clone(),
                fingerprint: sdr,
                frequency: 0.0,
                pos_tags: Vec::new(),
            };

            fingerprints.insert(word, wf);
        }

        // Read inverted index if present
        let inverted_index = if header.has_inverted_index() {
            file.seek(SeekFrom::Start(header.inverted_index_offset))?;
            let mut index_data = Vec::new();
            file.read_to_end(&mut index_data)?;
            Some(bincode::deserialize(&index_data)?)
        } else {
            None
        };

        Ok((header, fingerprints, inverted_index))
    }

    /// Memory-maps a retina file for fast access.
    pub fn mmap<P: AsRef<Path>>(path: P) -> Result<MmappedRetina> {
        MmappedRetina::open(path)
    }
}

/// A memory-mapped retina for instant loading.
///
/// Uses binary search on sorted words for O(log n) lookup without
/// building a HashMap. The inverted index is accessed lazily.
pub struct MmappedRetina {
    mmap: Mmap,
    header: RetinaHeader,
}

impl MmappedRetina {
    /// Opens a retina file and memory-maps it.
    ///
    /// This is O(1) - no data structures are built at load time.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Parse header
        let header = RetinaHeader::from_bytes(&mmap[0..HEADER_SIZE])?;

        Ok(Self { mmap, header })
    }

    /// Gets the word at a given index.
    fn get_word_at_index(&self, idx: usize) -> Option<String> {
        if idx >= self.header.num_words as usize {
            return None;
        }

        let index_start = self.header.index_offset as usize;
        let entry_start = index_start + idx * IndexEntry::SIZE;
        let entry = IndexEntry::from_bytes(&self.mmap[entry_start..entry_start + IndexEntry::SIZE]);

        let str_start = self.header.string_pool_offset as usize + entry.string_offset as usize;
        let str_end = str_start + entry.string_len as usize;

        Some(String::from_utf8_lossy(&self.mmap[str_start..str_end]).to_string())
    }

    /// Binary search for a word in the sorted index.
    fn find_word_index(&self, word: &str) -> Option<usize> {
        let num_words = self.header.num_words as usize;
        if num_words == 0 {
            return None;
        }

        let mut left = 0;
        let mut right = num_words;

        while left < right {
            let mid = left + (right - left) / 2;
            let mid_word = self.get_word_at_index(mid)?;

            match mid_word.as_str().cmp(word) {
                std::cmp::Ordering::Equal => return Some(mid),
                std::cmp::Ordering::Less => left = mid + 1,
                std::cmp::Ordering::Greater => right = mid,
            }
        }

        None
    }

    /// Returns the header.
    pub fn header(&self) -> &RetinaHeader {
        &self.header
    }

    /// Returns the number of words.
    pub fn len(&self) -> usize {
        self.header.num_words as usize
    }

    /// Checks if the retina is empty.
    pub fn is_empty(&self) -> bool {
        self.header.num_words == 0
    }

    /// Checks if a word exists.
    pub fn contains(&self, word: &str) -> bool {
        self.find_word_index(word).is_some()
    }

    /// Gets a word's fingerprint.
    pub fn get(&self, word: &str) -> Option<Sdr> {
        let idx = self.find_word_index(word)?;
        self.get_by_index(idx)
    }

    /// Gets a fingerprint by index.
    pub fn get_by_index(&self, idx: usize) -> Option<Sdr> {
        if idx >= self.header.num_words as usize {
            return None;
        }

        let index_start = self.header.index_offset as usize;
        let entry_start = index_start + idx * IndexEntry::SIZE;
        let entry = IndexEntry::from_bytes(&self.mmap[entry_start..entry_start + IndexEntry::SIZE]);

        let fp_start = self.header.fingerprint_offset as usize + entry.fp_offset as usize;
        let fp_end = fp_start + entry.fp_len as usize;

        let bitmap = crate::roaring::RoaringBitmap::deserialize_unchecked_from(&self.mmap[fp_start..fp_end])
            .ok()?;

        let grid_size = self.header.dimension * self.header.dimension;
        Some(Sdr::from_bitmap(bitmap, grid_size))
    }

    /// Gets a word by index.
    pub fn get_word(&self, idx: usize) -> Option<String> {
        if idx >= self.header.num_words as usize {
            return None;
        }

        let index_start = self.header.index_offset as usize;
        let entry_start = index_start + idx * IndexEntry::SIZE;
        let entry = IndexEntry::from_bytes(&self.mmap[entry_start..entry_start + IndexEntry::SIZE]);

        let str_start = self.header.string_pool_offset as usize + entry.string_offset as usize;
        let str_end = str_start + entry.string_len as usize;

        Some(String::from_utf8_lossy(&self.mmap[str_start..str_end]).to_string())
    }

    /// Returns the number of words (vocabulary size).
    pub fn num_words(&self) -> usize {
        self.header.num_words as usize
    }

    /// Returns true if the retina has an inverted index for fast similarity search.
    pub fn has_inverted_index(&self) -> bool {
        self.header.has_mmap_index() || self.header.has_inverted_index()
    }

    /// Gets a posting list from the mmap-friendly index (lazy loading).
    fn get_posting_list(&self, position: u32) -> Option<crate::roaring::RoaringBitmap> {
        if !self.header.has_mmap_index() {
            return None;
        }

        let grid_size = self.header.dimension * self.header.dimension;
        if position >= grid_size {
            return None;
        }

        // The mmap-friendly index starts at inverted_index_offset
        // Format: offset_table (grid_size * 8 bytes) + posting_data
        let index_start = self.header.inverted_index_offset as usize;
        let entry_offset = index_start + (position as usize) * 8;

        if entry_offset + 8 > self.mmap.len() {
            return None;
        }

        let offset = u32::from_le_bytes([
            self.mmap[entry_offset],
            self.mmap[entry_offset + 1],
            self.mmap[entry_offset + 2],
            self.mmap[entry_offset + 3],
        ]) as usize;
        let len = u32::from_le_bytes([
            self.mmap[entry_offset + 4],
            self.mmap[entry_offset + 5],
            self.mmap[entry_offset + 6],
            self.mmap[entry_offset + 7],
        ]) as usize;

        if len == 0 {
            return Some(crate::roaring::RoaringBitmap::new());
        }

        // Data starts after the offset table
        let data_offset = index_start + (grid_size as usize) * 8;
        let data_start = data_offset + offset;
        let data_end = data_start + len;

        if data_end > self.mmap.len() {
            return None;
        }

        crate::roaring::RoaringBitmap::deserialize_unchecked_from(&self.mmap[data_start..data_end]).ok()
    }

    /// Finds words similar to a given word using the inverted index.
    ///
    /// Returns up to `k` (word, similarity) pairs sorted by similarity.
    /// Returns None if the word is not found or no inverted index is available.
    pub fn find_similar(&self, word: &str, k: usize) -> Option<Vec<(String, f64)>> {
        if !self.header.has_mmap_index() {
            return None;
        }

        let target_fp = self.get(word)?;

        // Search using lazy loading of posting lists
        let mut word_scores: HashMap<u32, u32> = HashMap::new();

        for pos in target_fp.iter() {
            if let Some(posting_list) = self.get_posting_list(pos) {
                for word_idx in posting_list.iter() {
                    *word_scores.entry(word_idx).or_insert(0) += 1;
                }
            }
        }

        // Get top candidates by overlap count
        let mut sorted: Vec<(u32, u32)> = word_scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(k * 10);

        // Compute actual cosine similarity for candidates
        let mut results: Vec<(String, f64)> = sorted
            .into_iter()
            .filter_map(|(word_idx, _)| {
                let candidate_word = self.get_word_at_index(word_idx as usize)?;
                if candidate_word == word {
                    return None;
                }
                let fp = self.get_by_index(word_idx as usize)?;
                let sim = target_fp.cosine_similarity(&fp);
                Some((candidate_word, sim))
            })
            .collect();

        // Sort by similarity descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        Some(results)
    }

    /// Returns the grid dimension.
    pub fn dimension(&self) -> u32 {
        self.header.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_fingerprints() -> HashMap<String, WordFingerprint> {
        let mut fps = HashMap::new();
        fps.insert(
            "hello".to_string(),
            WordFingerprint::new("hello".to_string(), &[1, 5, 10, 100], 16384),
        );
        fps.insert(
            "world".to_string(),
            WordFingerprint::new("world".to_string(), &[2, 6, 20, 200], 16384),
        );
        fps.insert(
            "test".to_string(),
            WordFingerprint::new("test".to_string(), &[3, 7, 30, 300], 16384),
        );
        fps
    }

    #[test]
    fn test_header_roundtrip() {
        let header = RetinaHeader::new(128, 1000);
        let bytes = header.to_bytes();
        let recovered = RetinaHeader::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.dimension, 128);
        assert_eq!(recovered.num_words, 1000);
        assert_eq!(recovered.version, VERSION);
    }

    #[test]
    fn test_write_and_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.retina");

        let fps = create_test_fingerprints();
        RetinaFormat::write(&path, &fps, 128).unwrap();

        let (header, recovered) = RetinaFormat::read(&path).unwrap();

        assert_eq!(header.dimension, 128);
        assert_eq!(header.num_words, 3);
        assert_eq!(recovered.len(), 3);

        let hello = recovered.get("hello").unwrap();
        assert!(hello.fingerprint.contains(1));
        assert!(hello.fingerprint.contains(100));
    }

    #[test]
    fn test_mmap() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.retina");

        let fps = create_test_fingerprints();
        RetinaFormat::write(&path, &fps, 128).unwrap();

        let mmap = RetinaFormat::mmap(&path).unwrap();

        assert_eq!(mmap.len(), 3);
        assert!(mmap.contains("hello"));
        assert!(mmap.contains("world"));

        let hello = mmap.get("hello").unwrap();
        assert!(hello.contains(1));
        assert!(hello.contains(100));
    }
}
