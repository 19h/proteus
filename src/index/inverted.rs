//! Inverted index for efficient similarity search.

use crate::fingerprint::{Sdr, WordFingerprint};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Inverted index mapping positions to words.
///
/// This enables fast approximate nearest neighbor search by finding
/// words that share positions with a query fingerprint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvertedIndex {
    /// Maps position -> set of word indices that have this position active.
    posting_lists: Vec<RoaringBitmap>,
    /// Maps word index -> word string.
    index_to_word: Vec<String>,
    /// Maps word string -> word index.
    word_to_index: HashMap<String, u32>,
    /// Grid size.
    grid_size: u32,
}

impl InvertedIndex {
    /// Creates an empty inverted index.
    pub fn new(grid_size: u32) -> Self {
        let posting_lists = (0..grid_size)
            .map(|_| RoaringBitmap::new())
            .collect();

        Self {
            posting_lists,
            index_to_word: Vec::new(),
            word_to_index: HashMap::new(),
            grid_size,
        }
    }

    /// Creates an inverted index from word fingerprints.
    pub fn from_fingerprints(
        fingerprints: &HashMap<String, WordFingerprint>,
        grid_size: u32,
    ) -> Self {
        let mut index = Self::new(grid_size);

        // Sort words for consistent ordering
        let mut sorted_words: Vec<&String> = fingerprints.keys().collect();
        sorted_words.sort();

        for word in sorted_words {
            let wf = &fingerprints[word];
            index.insert(word, &wf.fingerprint);
        }

        index
    }

    /// Inserts a word with its fingerprint into the index.
    pub fn insert(&mut self, word: &str, fingerprint: &Sdr) {
        let word_idx = self.get_or_create_word_index(word);

        for pos in fingerprint.iter() {
            if (pos as usize) < self.posting_lists.len() {
                self.posting_lists[pos as usize].insert(word_idx);
            }
        }
    }

    /// Gets or creates an index for a word.
    fn get_or_create_word_index(&mut self, word: &str) -> u32 {
        if let Some(&idx) = self.word_to_index.get(word) {
            idx
        } else {
            let idx = self.index_to_word.len() as u32;
            self.index_to_word.push(word.to_string());
            self.word_to_index.insert(word.to_string(), idx);
            idx
        }
    }

    /// Removes a word from the index.
    pub fn remove(&mut self, word: &str, fingerprint: &Sdr) {
        if let Some(&word_idx) = self.word_to_index.get(word) {
            for pos in fingerprint.iter() {
                if (pos as usize) < self.posting_lists.len() {
                    self.posting_lists[pos as usize].remove(word_idx);
                }
            }
        }
    }

    /// Searches for words similar to a query fingerprint.
    ///
    /// Returns up to `k` words that share the most positions with the query.
    pub fn search(&self, query: &Sdr, k: usize) -> Vec<String> {
        let candidates = self.search_with_scores(query, k);
        candidates.into_iter().map(|(word, _)| word).collect()
    }

    /// Searches for words with similarity scores.
    ///
    /// Returns up to `k` (word, overlap_count) pairs.
    pub fn search_with_scores(&self, query: &Sdr, k: usize) -> Vec<(String, u32)> {
        // Count how many query positions each word shares
        let mut word_scores: HashMap<u32, u32> = HashMap::new();

        for pos in query.iter() {
            if (pos as usize) < self.posting_lists.len() {
                for word_idx in self.posting_lists[pos as usize].iter() {
                    *word_scores.entry(word_idx).or_insert(0) += 1;
                }
            }
        }

        // Sort by score and take top k
        let mut sorted: Vec<(u32, u32)> = word_scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(k);

        sorted
            .into_iter()
            .filter_map(|(word_idx, score)| {
                self.index_to_word
                    .get(word_idx as usize)
                    .map(|word| (word.clone(), score))
            })
            .collect()
    }

    /// Finds all words that have a specific position active.
    pub fn words_at_position(&self, position: u32) -> Vec<String> {
        if (position as usize) >= self.posting_lists.len() {
            return Vec::new();
        }

        self.posting_lists[position as usize]
            .iter()
            .filter_map(|idx| self.index_to_word.get(idx as usize).cloned())
            .collect()
    }

    /// Returns the number of words in the index.
    pub fn len(&self) -> usize {
        self.index_to_word.len()
    }

    /// Checks if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.index_to_word.is_empty()
    }

    /// Returns the grid size.
    pub fn grid_size(&self) -> u32 {
        self.grid_size
    }

    /// Returns statistics about the index.
    pub fn stats(&self) -> IndexStats {
        let total_entries: u64 = self.posting_lists.iter().map(|pl| pl.len()).sum();
        let non_empty_lists = self.posting_lists.iter().filter(|pl| !pl.is_empty()).count();
        let max_list_size = self.posting_lists.iter().map(|pl| pl.len()).max().unwrap_or(0);
        let avg_list_size = if non_empty_lists > 0 {
            total_entries as f64 / non_empty_lists as f64
        } else {
            0.0
        };

        IndexStats {
            num_words: self.index_to_word.len(),
            num_positions: self.grid_size as usize,
            non_empty_positions: non_empty_lists,
            total_entries: total_entries as usize,
            max_list_size: max_list_size as usize,
            avg_list_size,
        }
    }

    /// Returns posting list sizes for analysis.
    pub fn posting_list_sizes(&self) -> Vec<usize> {
        self.posting_lists.iter().map(|pl| pl.len() as usize).collect()
    }
}

/// Statistics about the inverted index.
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// Number of words in the index.
    pub num_words: usize,
    /// Number of positions (grid size).
    pub num_positions: usize,
    /// Number of positions with at least one word.
    pub non_empty_positions: usize,
    /// Total number of (position, word) entries.
    pub total_entries: usize,
    /// Size of the largest posting list.
    pub max_list_size: usize,
    /// Average posting list size (for non-empty lists).
    pub avg_list_size: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_fingerprints() -> HashMap<String, WordFingerprint> {
        let mut fps = HashMap::new();
        fps.insert(
            "hello".to_string(),
            WordFingerprint::new("hello".to_string(), &[1, 2, 3, 10, 20], 100),
        );
        fps.insert(
            "world".to_string(),
            WordFingerprint::new("world".to_string(), &[2, 3, 4, 20, 30], 100),
        );
        fps.insert(
            "test".to_string(),
            WordFingerprint::new("test".to_string(), &[5, 6, 7, 50, 60], 100),
        );
        fps
    }

    #[test]
    fn test_index_creation() {
        let fps = create_test_fingerprints();
        let index = InvertedIndex::from_fingerprints(&fps, 100);

        assert_eq!(index.len(), 3);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_search() {
        let fps = create_test_fingerprints();
        let index = InvertedIndex::from_fingerprints(&fps, 100);

        // Search with a query that overlaps with "hello" and "world"
        let query = Sdr::from_positions(&[2, 3, 10], 100);
        let results = index.search(&query, 2);

        assert_eq!(results.len(), 2);
        // Both hello and world share positions 2 and 3
        assert!(results.contains(&"hello".to_string()));
        assert!(results.contains(&"world".to_string()));
    }

    #[test]
    fn test_search_with_scores() {
        let fps = create_test_fingerprints();
        let index = InvertedIndex::from_fingerprints(&fps, 100);

        let query = Sdr::from_positions(&[1, 2, 3, 10], 100);
        let results = index.search_with_scores(&query, 3);

        // "hello" has positions 1, 2, 3, 10 - all 4 match
        // "world" has positions 2, 3 - only 2 match
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "hello");
        assert_eq!(results[0].1, 4); // 4 positions overlap
    }

    #[test]
    fn test_words_at_position() {
        let fps = create_test_fingerprints();
        let index = InvertedIndex::from_fingerprints(&fps, 100);

        let words = index.words_at_position(2);
        assert_eq!(words.len(), 2); // hello and world both have position 2

        let words = index.words_at_position(1);
        assert_eq!(words.len(), 1); // only hello has position 1
    }

    #[test]
    fn test_insert_remove() {
        let mut index = InvertedIndex::new(100);

        let fp = Sdr::from_positions(&[1, 2, 3], 100);
        index.insert("test", &fp);

        assert_eq!(index.len(), 1);
        assert_eq!(index.words_at_position(1).len(), 1);

        index.remove("test", &fp);
        assert_eq!(index.words_at_position(1).len(), 0);
    }

    #[test]
    fn test_stats() {
        let fps = create_test_fingerprints();
        let index = InvertedIndex::from_fingerprints(&fps, 100);
        let stats = index.stats();

        assert_eq!(stats.num_words, 3);
        assert_eq!(stats.num_positions, 100);
        assert!(stats.non_empty_positions > 0);
    }
}
