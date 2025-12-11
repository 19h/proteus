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
//! ```
//!
//! ### Header (64 bytes)
//! - Magic number (4 bytes): "PRET"
//! - Version (2 bytes)
//! - Flags (2 bytes)
//! - Grid dimension (4 bytes)
//! - Number of words (4 bytes)
//! - Index table offset (8 bytes)
//! - Fingerprint data offset (8 bytes)
//! - String pool offset (8 bytes)
//! - Reserved (24 bytes)
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
use memmap2::{Mmap, MmapOptions};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Magic number for Proteus retina files.
const MAGIC: &[u8; 4] = b"PRET";

/// Current format version.
const VERSION: u16 = 1;

/// Header size in bytes.
const HEADER_SIZE: usize = 64;

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
    /// Format version.
    pub version: u16,
    /// Flags.
    pub flags: u16,
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
            version: VERSION,
            flags: 0,
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

        // Reserved (bytes 40-63)
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

        Ok(Self {
            dimension,
            num_words,
            index_offset,
            fingerprint_offset,
            string_pool_offset,
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
    /// Writes fingerprints to a binary file.
    pub fn write<P: AsRef<Path>>(
        path: P,
        fingerprints: &HashMap<String, WordFingerprint>,
        dimension: u32,
    ) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let num_words = fingerprints.len() as u32;
        let mut header = RetinaHeader::new(dimension, num_words);

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

        writer.flush()?;
        Ok(())
    }

    /// Reads fingerprints from a binary file.
    pub fn read<P: AsRef<Path>>(path: P) -> Result<(RetinaHeader, HashMap<String, WordFingerprint>)> {
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

        // Read string pool
        file.seek(SeekFrom::Start(header.string_pool_offset))?;
        let mut string_pool = Vec::new();
        file.read_to_end(&mut string_pool)?;

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

        Ok((header, fingerprints))
    }

    /// Memory-maps a retina file for fast access.
    pub fn mmap<P: AsRef<Path>>(path: P) -> Result<MmappedRetina> {
        MmappedRetina::open(path)
    }
}

/// A memory-mapped retina for fast access.
pub struct MmappedRetina {
    mmap: Mmap,
    header: RetinaHeader,
    word_index: HashMap<String, usize>,
}

impl MmappedRetina {
    /// Opens a retina file and memory-maps it.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Parse header
        let header = RetinaHeader::from_bytes(&mmap[0..HEADER_SIZE])?;

        // Build word index
        let mut word_index = HashMap::with_capacity(header.num_words as usize);
        let index_start = header.index_offset as usize;

        for i in 0..header.num_words as usize {
            let entry_start = index_start + i * IndexEntry::SIZE;
            let entry = IndexEntry::from_bytes(&mmap[entry_start..entry_start + IndexEntry::SIZE]);

            let str_start = header.string_pool_offset as usize + entry.string_offset as usize;
            let str_end = str_start + entry.string_len as usize;
            let word = String::from_utf8_lossy(&mmap[str_start..str_end]).to_string();

            word_index.insert(word, i);
        }

        Ok(Self {
            mmap,
            header,
            word_index,
        })
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
        self.word_index.contains_key(word)
    }

    /// Gets a word's fingerprint.
    pub fn get(&self, word: &str) -> Option<Sdr> {
        let idx = *self.word_index.get(word)?;
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

    /// Returns an iterator over all words.
    pub fn words(&self) -> impl Iterator<Item = &String> {
        self.word_index.keys()
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
