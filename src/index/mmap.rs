//! Memory-mappable inverted index for instant loading.
//!
//! This format allows the inverted index to be memory-mapped and accessed
//! without deserializing the entire structure upfront. Only the posting lists
//! that are actually accessed during a search are deserialized on-demand.
//!
//! ## Format Layout
//!
//! ```text
//! +------------------------+
//! | Header (16 bytes)      |
//! +------------------------+
//! | Offset Table           |
//! | (grid_size * 8 bytes)  |
//! +------------------------+
//! | Posting List Data      |
//! | (variable)             |
//! +------------------------+
//! ```

use crate::error::{ProteusError, Result};
use crate::fingerprint::Sdr;
use crate::roaring::RoaringBitmap;
use memmap2::{Mmap, MmapOptions};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const MMAP_INDEX_MAGIC: &[u8; 4] = b"PIDX";
const MMAP_INDEX_VERSION: u32 = 1;
const HEADER_SIZE: usize = 16;

/// Memory-mapped inverted index for instant loading.
///
/// Posting lists are deserialized on-demand during search, making
/// initial load time O(1) regardless of vocabulary size.
pub struct MmappedInvertedIndex {
    mmap: Mmap,
    grid_size: u32,
    data_offset: u64,
}

impl MmappedInvertedIndex {
    /// Opens a memory-mapped inverted index file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(ProteusError::InvalidRetinaFormat(
                "Index file too small".to_string(),
            ));
        }

        // Verify magic
        if &mmap[0..4] != MMAP_INDEX_MAGIC {
            return Err(ProteusError::InvalidRetinaFormat(
                "Invalid index magic number".to_string(),
            ));
        }

        let version = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]);
        if version != MMAP_INDEX_VERSION {
            return Err(ProteusError::InvalidRetinaFormat(format!(
                "Unsupported index version: {}",
                version
            )));
        }

        let grid_size = u32::from_le_bytes([mmap[8], mmap[9], mmap[10], mmap[11]]);
        let data_offset = u64::from_le_bytes([
            mmap[12], mmap[13], mmap[14], mmap[15], mmap[16], mmap[17], mmap[18], mmap[19],
        ]);

        Ok(Self {
            mmap,
            grid_size,
            data_offset,
        })
    }

    /// Returns the grid size.
    pub fn grid_size(&self) -> u32 {
        self.grid_size
    }

    /// Gets the posting list for a position (on-demand deserialization).
    fn get_posting_list(&self, position: u32) -> Option<RoaringBitmap> {
        if position >= self.grid_size {
            return None;
        }

        // Read offset table entry: 8 bytes per position (offset: u32, len: u32)
        // Header is 20 bytes (magic 4 + version 4 + grid_size 4 + data_offset 8)
        let entry_offset = 20 + (position as usize) * 8;
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
            return Some(RoaringBitmap::new());
        }

        let data_start = self.data_offset as usize + offset;
        let data_end = data_start + len;

        if data_end > self.mmap.len() {
            return None;
        }

        RoaringBitmap::deserialize_unchecked_from(&self.mmap[data_start..data_end]).ok()
    }

    /// Searches for words similar to a query fingerprint.
    ///
    /// Returns up to `k` word indices with their overlap counts.
    pub fn search_with_scores(&self, query: &Sdr, k: usize) -> Vec<(u32, u32)> {
        let mut word_scores: HashMap<u32, u32> = HashMap::new();

        for pos in query.iter() {
            if let Some(posting_list) = self.get_posting_list(pos) {
                for word_idx in posting_list.iter() {
                    *word_scores.entry(word_idx).or_insert(0) += 1;
                }
            }
        }

        // Sort by score and take top k
        let mut sorted: Vec<(u32, u32)> = word_scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(k);

        sorted
    }
}

/// Writer for creating memory-mapped inverted index files.
pub struct MmappedIndexWriter;

impl MmappedIndexWriter {
    /// Writes an inverted index to a memory-mappable file.
    pub fn write<P: AsRef<Path>>(
        path: P,
        posting_lists: &[RoaringBitmap],
        grid_size: u32,
    ) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // First pass: serialize all posting lists and collect offsets
        let mut serialized: Vec<Vec<u8>> = Vec::with_capacity(grid_size as usize);
        let mut offsets: Vec<(u32, u32)> = Vec::with_capacity(grid_size as usize);
        let mut current_offset: u32 = 0;

        for i in 0..grid_size as usize {
            let bitmap = if i < posting_lists.len() {
                &posting_lists[i]
            } else {
                // Empty bitmap for missing positions
                offsets.push((0, 0));
                serialized.push(Vec::new());
                continue;
            };

            if bitmap.is_empty() {
                offsets.push((0, 0));
                serialized.push(Vec::new());
            } else {
                let mut data = Vec::new();
                bitmap
                    .serialize_into(&mut data)
                    .map_err(|e| ProteusError::Serialization(e.to_string()))?;
                let len = data.len() as u32;
                offsets.push((current_offset, len));
                current_offset += len;
                serialized.push(data);
            }
        }

        // Calculate data offset (after header + offset table)
        let offset_table_size = (grid_size as usize) * 8;
        let data_offset = (20 + offset_table_size) as u64;

        // Write header (20 bytes)
        writer.write_all(MMAP_INDEX_MAGIC)?;
        writer.write_all(&MMAP_INDEX_VERSION.to_le_bytes())?;
        writer.write_all(&grid_size.to_le_bytes())?;
        writer.write_all(&data_offset.to_le_bytes())?;

        // Write offset table
        for (offset, len) in &offsets {
            writer.write_all(&offset.to_le_bytes())?;
            writer.write_all(&len.to_le_bytes())?;
        }

        // Write posting list data
        for data in &serialized {
            if !data.is_empty() {
                writer.write_all(data)?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Creates a memory-mappable index from word fingerprints.
    pub fn from_fingerprints<P: AsRef<Path>>(
        path: P,
        fingerprints: &HashMap<String, crate::fingerprint::WordFingerprint>,
        grid_size: u32,
    ) -> Result<()> {
        use rayon::prelude::*;
        use std::sync::Mutex;

        // Sort words for consistent ordering (same as retina file)
        let mut sorted_words: Vec<&String> = fingerprints.keys().collect();
        sorted_words.sort();

        // Build posting lists in parallel
        let posting_lists: Vec<Mutex<RoaringBitmap>> = (0..grid_size)
            .map(|_| Mutex::new(RoaringBitmap::new()))
            .collect();

        sorted_words
            .par_iter()
            .enumerate()
            .for_each(|(word_idx, word)| {
                let wf = &fingerprints[*word];
                for pos in wf.fingerprint.iter() {
                    if (pos as usize) < posting_lists.len() {
                        posting_lists[pos as usize]
                            .lock()
                            .unwrap()
                            .insert(word_idx as u32);
                    }
                }
            });

        // Extract bitmaps
        let posting_lists: Vec<RoaringBitmap> = posting_lists
            .into_iter()
            .map(|m| m.into_inner().unwrap())
            .collect();

        Self::write(path, &posting_lists, grid_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_mmap_index_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.pidx");

        // Create test posting lists
        let mut posting_lists = vec![RoaringBitmap::new(); 100];
        posting_lists[5].insert(0);
        posting_lists[5].insert(1);
        posting_lists[10].insert(0);
        posting_lists[10].insert(2);
        posting_lists[20].insert(1);
        posting_lists[20].insert(2);

        // Write
        MmappedIndexWriter::write(&path, &posting_lists, 100).unwrap();

        // Read and verify
        let index = MmappedInvertedIndex::open(&path).unwrap();
        assert_eq!(index.grid_size(), 100);

        // Test search with a query that matches positions 5 and 10
        let query = Sdr::from_positions(&[5, 10], 100);
        let results = index.search_with_scores(&query, 10);

        // Word 0 should have score 2 (appears in both positions 5 and 10)
        assert!(results.iter().any(|(idx, score)| *idx == 0 && *score == 2));
    }
}
