//! Hyperdimensional Computing (HDC) / Vector Symbolic Architectures (VSA).
//!
//! This module implements operations for sparse binary distributed representations
//! based on hyperdimensional computing principles. These representations are:
//!
//! - **High-dimensional**: Typically 1000-10000 dimensions
//! - **Sparse**: Only a small fraction of bits are active (~2%)
//! - **Holographic**: Information is distributed across all dimensions
//! - **Robust**: Graceful degradation with noise or missing information
//!
//! ## Key Operations
//!
//! 1. **Bundling (⊕)**: Combines multiple hypervectors into one (set union)
//! 2. **Binding (⊗)**: Creates associations between hypervectors (XOR)
//! 3. **Permutation (ρ)**: Creates sequence/position encoding (rotate)
//! 4. **Similarity**: Measures overlap between hypervectors
//!
//! ## Applications
//!
//! - Compositional semantics (word ⊗ context)
//! - Sequence encoding (ρ^n(word))
//! - Analogical reasoning (king - man + woman ≈ queen)
//! - Memory retrieval (content-addressable lookup)
//!
//! References:
//! - Kanerva (2009): "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors"
//! - Gayler (2003): "Vector Symbolic Architectures Answer Jackendoff's Challenges for Cognitive Neuroscience"

use crate::fingerprint::Sdr;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

/// A hypervector in sparse binary format.
///
/// This is a thin wrapper around SDR with HDC-specific operations.
#[derive(Debug, Clone)]
pub struct Hypervector {
    /// The underlying sparse distributed representation.
    pub sdr: Sdr,
    /// Dimension (total number of possible positions).
    pub dimension: u32,
}

impl Hypervector {
    /// Create a new empty hypervector.
    pub fn new(dimension: u32) -> Self {
        Self {
            sdr: Sdr::new(dimension),
            dimension,
        }
    }

    /// Create a hypervector from an existing SDR.
    pub fn from_sdr(sdr: Sdr, dimension: u32) -> Self {
        Self { sdr, dimension }
    }

    /// Create a random hypervector with specified sparsity.
    pub fn random(dimension: u32, sparsity: f64, seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };

        let num_active = (dimension as f64 * sparsity) as usize;
        let mut positions: Vec<u32> = (0..dimension).collect();

        // Fisher-Yates shuffle for first num_active elements
        for i in 0..num_active.min(positions.len()) {
            let j = rng.gen_range(i..positions.len());
            positions.swap(i, j);
        }

        let active_positions: Vec<u32> = positions.into_iter().take(num_active).collect();
        let sdr = Sdr::from_positions(&active_positions, dimension);

        Self { sdr, dimension }
    }

    /// Create a hypervector with specified positions active.
    pub fn from_positions(positions: &[u32], dimension: u32) -> Self {
        Self {
            sdr: Sdr::from_positions(positions, dimension),
            dimension,
        }
    }

    /// Get the number of active bits.
    pub fn cardinality(&self) -> u64 {
        self.sdr.cardinality()
    }

    /// Get the sparsity (fraction of active bits).
    pub fn sparsity(&self) -> f64 {
        self.sdr.cardinality() as f64 / self.dimension as f64
    }

    /// Check if a position is active.
    pub fn contains(&self, position: u32) -> bool {
        self.sdr.contains(position)
    }

    /// Get all active positions.
    pub fn positions(&self) -> impl Iterator<Item = u32> + '_ {
        self.sdr.to_positions().into_iter()
    }

    // === HDC Operations ===

    /// **Bundling (⊕)**: Combine multiple hypervectors into a single one.
    ///
    /// For sparse binary vectors, this is typically implemented as:
    /// 1. Count votes for each position across all input vectors
    /// 2. Keep positions that appear in majority (or above threshold)
    ///
    /// This is a "soft union" that maintains sparsity.
    pub fn bundle(vectors: &[&Hypervector], threshold_frac: f64) -> Self {
        if vectors.is_empty() {
            return Self::new(0);
        }

        let dimension = vectors[0].dimension;
        let threshold = (vectors.len() as f64 * threshold_frac).ceil() as usize;

        // Count votes for each position
        let mut votes: HashMap<u32, usize> = HashMap::new();
        for hv in vectors {
            for pos in hv.positions() {
                *votes.entry(pos).or_insert(0) += 1;
            }
        }

        // Keep positions above threshold
        let positions: Vec<u32> = votes
            .into_iter()
            .filter(|&(_, count)| count >= threshold)
            .map(|(pos, _)| pos)
            .collect();

        Self::from_positions(&positions, dimension)
    }

    /// **Bundling** with automatic threshold (majority voting).
    pub fn bundle_majority(vectors: &[&Hypervector]) -> Self {
        Self::bundle(vectors, 0.5)
    }

    /// **Weighted Bundling**: Bundle with weights for each vector.
    pub fn bundle_weighted(vectors: &[(&Hypervector, f64)]) -> Self {
        if vectors.is_empty() {
            return Self::new(0);
        }

        let dimension = vectors[0].0.dimension;
        let total_weight: f64 = vectors.iter().map(|(_, w)| w).sum();
        let threshold = total_weight * 0.5;

        let mut votes: HashMap<u32, f64> = HashMap::new();
        for (hv, weight) in vectors {
            for pos in hv.positions() {
                *votes.entry(pos).or_insert(0.0) += weight;
            }
        }

        let positions: Vec<u32> = votes
            .into_iter()
            .filter(|&(_, vote)| vote >= threshold)
            .map(|(pos, _)| pos)
            .collect();

        Self::from_positions(&positions, dimension)
    }

    /// **Binding (⊗)**: Create association between two hypervectors.
    ///
    /// For sparse binary vectors, binding is typically XOR.
    /// The result is dissimilar to both inputs but can be "unbound"
    /// by XORing with either input to recover (approximately) the other.
    pub fn bind(&self, other: &Hypervector) -> Self {
        let xored = self.sdr.xor(&other.sdr);
        Self::from_sdr(xored, self.dimension)
    }

    /// **Unbind**: Inverse of binding (also XOR for binary vectors).
    pub fn unbind(&self, other: &Hypervector) -> Self {
        self.bind(other) // XOR is self-inverse
    }

    /// **Permutation (ρ)**: Rotate positions by an amount.
    ///
    /// Used for encoding sequences: ρ^n(x) encodes x at position n.
    pub fn permute(&self, amount: i32) -> Self {
        let dim = self.dimension as i32;
        let positions: Vec<u32> = self
            .positions()
            .map(|p| ((p as i32 + amount).rem_euclid(dim)) as u32)
            .collect();

        Self::from_positions(&positions, self.dimension)
    }

    /// **Inverse Permutation (ρ^-1)**: Rotate in the opposite direction.
    pub fn permute_inverse(&self, amount: i32) -> Self {
        self.permute(-amount)
    }

    /// **Similarity**: Compute overlap-based similarity (normalized by minimum cardinality).
    pub fn similarity(&self, other: &Hypervector) -> f64 {
        self.sdr.overlap_similarity(&other.sdr)
    }

    /// **Cosine Similarity**: Normalized dot product.
    pub fn cosine_similarity(&self, other: &Hypervector) -> f64 {
        self.sdr.cosine_similarity(&other.sdr)
    }

    /// **Jaccard Similarity**: Intersection over union.
    pub fn jaccard_similarity(&self, other: &Hypervector) -> f64 {
        self.sdr.jaccard_similarity(&other.sdr)
    }

    /// **Hamming Distance**: Number of differing positions.
    pub fn hamming_distance(&self, other: &Hypervector) -> u64 {
        self.sdr.xor(&other.sdr).cardinality()
    }

    /// **Sparsify**: Reduce to target number of active bits.
    ///
    /// Keeps the bits with highest "importance" (here: random selection).
    /// For weighted representations, would keep highest-weighted bits.
    pub fn sparsify(&self, target_bits: usize) -> Self {
        if self.cardinality() as usize <= target_bits {
            return self.clone();
        }

        let mut positions: Vec<u32> = self.positions().collect();

        // Random selection (could be improved with importance weights)
        use rand::seq::SliceRandom;
        let mut rng = ChaCha8Rng::from_entropy();
        positions.shuffle(&mut rng);
        positions.truncate(target_bits);

        Self::from_positions(&positions, self.dimension)
    }

    /// **Thin** to target sparsity.
    pub fn thin(&self, target_sparsity: f64) -> Self {
        let target_bits = (self.dimension as f64 * target_sparsity) as usize;
        self.sparsify(target_bits)
    }
}

/// An item memory (associative memory) for hypervectors.
///
/// Supports content-addressable retrieval: given a noisy or partial query,
/// retrieve the most similar stored hypervector.
pub struct ItemMemory {
    /// Stored items: name -> hypervector.
    items: HashMap<String, Hypervector>,
    /// Dimension of hypervectors.
    dimension: u32,
    /// Default sparsity for new items.
    sparsity: f64,
}

impl ItemMemory {
    /// Create a new item memory.
    pub fn new(dimension: u32, sparsity: f64) -> Self {
        Self {
            items: HashMap::new(),
            dimension,
            sparsity,
        }
    }

    /// Store an item with a name.
    pub fn store(&mut self, name: &str, item: Hypervector) {
        self.items.insert(name.to_string(), item);
    }

    /// Generate and store a random hypervector for an item.
    pub fn encode(&mut self, name: &str, seed: Option<u64>) -> Hypervector {
        let hv = Hypervector::random(self.dimension, self.sparsity, seed);
        self.items.insert(name.to_string(), hv.clone());
        hv
    }

    /// Get an item by name.
    pub fn get(&self, name: &str) -> Option<&Hypervector> {
        self.items.get(name)
    }

    /// Query the memory: find the most similar item to the query.
    pub fn query(&self, query: &Hypervector) -> Option<(&str, f64)> {
        self.items
            .iter()
            .map(|(name, hv)| (name.as_str(), query.similarity(hv)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }

    /// Query and return top-k matches.
    pub fn query_top_k(&self, query: &Hypervector, k: usize) -> Vec<(&str, f64)> {
        let mut results: Vec<_> = self
            .items
            .iter()
            .map(|(name, hv)| (name.as_str(), query.similarity(hv)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        results
    }

    /// Number of items in memory.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if memory is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// Sequence encoder using permutation-based encoding.
///
/// Encodes sequences as: S = ρ^(n-1)(x_1) ⊕ ρ^(n-2)(x_2) ⊕ ... ⊕ x_n
/// where ρ is permutation and ⊕ is bundling.
pub struct SequenceEncoder {
    /// Item memory for atomic symbols.
    memory: ItemMemory,
    /// Maximum sequence length.
    max_length: usize,
}

impl SequenceEncoder {
    /// Create a new sequence encoder.
    pub fn new(dimension: u32, sparsity: f64, max_length: usize) -> Self {
        Self {
            memory: ItemMemory::new(dimension, sparsity),
            max_length,
        }
    }

    /// Encode an item (get or create its hypervector).
    pub fn encode_item(&mut self, item: &str) -> Hypervector {
        if let Some(hv) = self.memory.get(item) {
            hv.clone()
        } else {
            self.memory.encode(item, None)
        }
    }

    /// Encode a sequence of items.
    pub fn encode_sequence(&mut self, items: &[&str]) -> Hypervector {
        if items.is_empty() {
            return Hypervector::new(self.memory.dimension);
        }

        let n = items.len().min(self.max_length);

        // Encode each item with position-based permutation
        let encoded: Vec<Hypervector> = items
            .iter()
            .take(n)
            .enumerate()
            .map(|(i, &item)| {
                let hv = self.encode_item(item);
                // Permute by position from the end
                let shift = (n - 1 - i) as i32;
                hv.permute(shift)
            })
            .collect();

        // Bundle all encoded items
        let refs: Vec<&Hypervector> = encoded.iter().collect();
        Hypervector::bundle(&refs, 0.5)
    }

    /// Decode: find items present in an encoded sequence.
    pub fn decode_sequence(&self, encoded: &Hypervector, threshold: f64) -> Vec<(String, usize, f64)> {
        let mut results = Vec::new();

        for (name, hv) in &self.memory.items {
            // Try each position
            for pos in 0..self.max_length {
                let unshifted = encoded.permute_inverse(pos as i32);
                let sim = unshifted.similarity(hv);

                if sim > threshold {
                    results.push((name.clone(), pos, sim));
                }
            }
        }

        // Sort by similarity descending
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        results
    }

    /// Get the item memory.
    pub fn memory(&self) -> &ItemMemory {
        &self.memory
    }
}

/// N-gram encoder for text.
///
/// Creates hypervector representations of n-grams by binding
/// consecutive character/word hypervectors.
pub struct NgramEncoder {
    /// Item memory for characters/tokens.
    memory: ItemMemory,
    /// N-gram size.
    n: usize,
}

impl NgramEncoder {
    /// Create a new n-gram encoder.
    pub fn new(dimension: u32, sparsity: f64, n: usize) -> Self {
        Self {
            memory: ItemMemory::new(dimension, sparsity),
            n,
        }
    }

    /// Encode a single n-gram.
    pub fn encode_ngram(&mut self, ngram: &[&str]) -> Hypervector {
        if ngram.is_empty() {
            return Hypervector::new(self.memory.dimension);
        }

        // Get hypervectors for each element
        let hvs: Vec<Hypervector> = ngram
            .iter()
            .map(|&item| {
                if let Some(hv) = self.memory.get(item) {
                    hv.clone()
                } else {
                    self.memory.encode(item, None)
                }
            })
            .collect();

        // Bind with position permutation
        let mut result = hvs[0].clone();
        for (i, hv) in hvs.iter().enumerate().skip(1) {
            let permuted = hv.permute(i as i32);
            result = result.bind(&permuted);
        }

        result
    }

    /// Encode text as bundle of all n-grams.
    pub fn encode_text(&mut self, tokens: &[&str]) -> Hypervector {
        if tokens.len() < self.n {
            return Hypervector::new(self.memory.dimension);
        }

        let ngrams: Vec<Hypervector> = (0..=tokens.len() - self.n)
            .map(|i| {
                let ngram: Vec<&str> = tokens[i..i + self.n].to_vec();
                self.encode_ngram(&ngram)
            })
            .collect();

        let refs: Vec<&Hypervector> = ngrams.iter().collect();
        Hypervector::bundle(&refs, 0.3)
    }
}

/// Analogy solver using HDC operations.
///
/// Solves analogies of the form: a is to b as c is to ?
/// Using: ? ≈ b - a + c (in vector space)
/// For HDC: ? ≈ unbind(b, a) ⊗ c
pub struct AnalogySolver {
    /// Item memory.
    memory: ItemMemory,
}

impl AnalogySolver {
    /// Create a new analogy solver.
    pub fn new(memory: ItemMemory) -> Self {
        Self { memory }
    }

    /// Solve an analogy: a is to b as c is to ?
    pub fn solve(&self, a: &str, b: &str, c: &str) -> Option<Vec<(&str, f64)>> {
        let hv_a = self.memory.get(a)?;
        let hv_b = self.memory.get(b)?;
        let hv_c = self.memory.get(c)?;

        // Compute the analogy vector: b ⊗ a^(-1) ⊗ c
        // For XOR-based binding, a^(-1) = a
        let relation = hv_b.unbind(hv_a);
        let query = relation.bind(hv_c);

        Some(self.memory.query_top_k(&query, 5))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypervector_random() {
        let hv = Hypervector::random(10000, 0.02, Some(42));
        let sparsity = hv.sparsity();
        assert!((sparsity - 0.02).abs() < 0.005);
    }

    #[test]
    fn test_bundle() {
        let hv1 = Hypervector::random(1000, 0.1, Some(1));
        let hv2 = Hypervector::random(1000, 0.1, Some(2));
        let hv3 = Hypervector::random(1000, 0.1, Some(3));

        let bundled = Hypervector::bundle(&[&hv1, &hv2, &hv3], 0.5);

        // Bundled should be similar to all inputs
        assert!(bundled.similarity(&hv1) > 0.3);
        assert!(bundled.similarity(&hv2) > 0.3);
        assert!(bundled.similarity(&hv3) > 0.3);
    }

    #[test]
    fn test_bind_unbind() {
        // Use 50% sparsity - this is the canonical density for HDC where
        // XOR binding produces pseudo-orthogonal (dissimilar) vectors.
        // With sparse vectors (10%), XOR preserves too much structure.
        let hv1 = Hypervector::random(1000, 0.5, Some(1));
        let hv2 = Hypervector::random(1000, 0.5, Some(2));

        let bound = hv1.bind(&hv2);

        // Bound should be dissimilar to both inputs (using Jaccard)
        // With 50% sparsity, XOR produces ~50% density with ~25% overlap
        assert!(bound.sdr.jaccard_similarity(&hv1.sdr) < 0.4);
        assert!(bound.sdr.jaccard_similarity(&hv2.sdr) < 0.4);

        // Unbinding should recover the other exactly (XOR is self-inverse)
        let recovered = bound.unbind(&hv1);
        assert_eq!(recovered.sdr.jaccard_similarity(&hv2.sdr), 1.0);
    }

    #[test]
    fn test_permutation() {
        let hv = Hypervector::random(1000, 0.1, Some(42));

        let permuted = hv.permute(100);
        let restored = permuted.permute_inverse(100);

        // Should be dissimilar after permutation
        assert!(hv.similarity(&permuted) < 0.3);

        // Should be restored after inverse
        assert_eq!(hv.similarity(&restored), 1.0);
    }

    #[test]
    fn test_item_memory() {
        let mut mem = ItemMemory::new(1000, 0.05);

        mem.encode("apple", Some(1));
        mem.encode("banana", Some(2));
        mem.encode("cherry", Some(3));

        let apple = mem.get("apple").unwrap().clone();

        // Query should find exact match
        let (name, sim) = mem.query(&apple).unwrap();
        assert_eq!(name, "apple");
        assert!(sim > 0.9);
    }

    #[test]
    fn test_sequence_encoder() {
        // Use 50% sparsity for HDC - with sparse vectors, bundling
        // loses too much information due to low bit overlap.
        let mut encoder = SequenceEncoder::new(1000, 0.5, 10);

        let seq1 = encoder.encode_sequence(&["the", "cat", "sat"]);
        let seq2 = encoder.encode_sequence(&["the", "dog", "sat"]);
        let seq3 = encoder.encode_sequence(&["a", "bird", "flew"]);

        // Similar sequences should have more similarity (2/3 words shared)
        // than completely different sequences (0/3 words shared)
        let sim_12 = seq1.sdr.jaccard_similarity(&seq2.sdr);
        let sim_13 = seq1.sdr.jaccard_similarity(&seq3.sdr);
        assert!(sim_12 > sim_13, "seq1~seq2 ({}) should be > seq1~seq3 ({})", sim_12, sim_13);
    }
}
