//! Locality-Sensitive Hashing (LSH) for fast approximate similarity search.
//!
//! This module implements several LSH schemes optimized for different similarity measures:
//!
//! 1. **SimHash**: For cosine/angular similarity on dense vectors
//! 2. **MinHash**: For Jaccard similarity on sets (SDRs)
//! 3. **BitSampling**: Ultra-fast LSH for binary vectors (SDRs)
//!
//! All schemes support multi-probe LSH for improved recall without increasing memory.
//!
//! References:
//! - Charikar (2002): "Similarity estimation techniques from rounding algorithms"
//! - Broder et al. (1997): "Syntactic clustering of the web"
//! - Indyk & Motwani (1998): "Approximate nearest neighbors: towards removing the curse of dimensionality"

use crate::fingerprint::Sdr;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{HashMap, HashSet};

/// SimHash for angular/cosine similarity on dense vectors.
///
/// Projects vectors onto random hyperplanes and takes the sign of each projection.
/// Hamming distance between SimHash signatures approximates angular distance.
pub struct SimHash {
    /// Random hyperplanes (one per bit).
    hyperplanes: Vec<Vec<f32>>,
    /// Number of bits in the signature.
    num_bits: usize,
}

impl SimHash {
    /// Create a new SimHash with specified signature size.
    pub fn new(dim: usize, num_bits: usize, seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };

        // Generate random hyperplanes from N(0,1)
        let hyperplanes: Vec<Vec<f32>> = (0..num_bits)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        // Box-Muller transform for normal distribution
                        let u1: f32 = rng.gen();
                        let u2: f32 = rng.gen();
                        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
                    })
                    .collect()
            })
            .collect();

        Self {
            hyperplanes,
            num_bits,
        }
    }

    /// Compute SimHash signature for a vector.
    pub fn hash(&self, vector: &[f32]) -> u64 {
        let mut signature = 0u64;

        for (i, hyperplane) in self.hyperplanes.iter().enumerate().take(64) {
            let dot: f32 = vector
                .iter()
                .zip(hyperplane.iter())
                .map(|(&a, &b)| a * b)
                .sum();

            if dot >= 0.0 {
                signature |= 1u64 << i;
            }
        }

        signature
    }

    /// Compute extended SimHash signature (arbitrary length).
    pub fn hash_extended(&self, vector: &[f32]) -> Vec<u64> {
        let num_words = (self.num_bits + 63) / 64;
        let mut signature = vec![0u64; num_words];

        for (i, hyperplane) in self.hyperplanes.iter().enumerate() {
            let dot: f32 = vector
                .iter()
                .zip(hyperplane.iter())
                .map(|(&a, &b)| a * b)
                .sum();

            if dot >= 0.0 {
                let word = i / 64;
                let bit = i % 64;
                signature[word] |= 1u64 << bit;
            }
        }

        signature
    }

    /// Compute Hamming distance between two signatures.
    #[inline]
    pub fn hamming_distance(a: u64, b: u64) -> u32 {
        (a ^ b).count_ones()
    }

    /// Estimate cosine similarity from Hamming distance.
    #[inline]
    pub fn estimate_cosine(hamming_dist: u32, num_bits: usize) -> f64 {
        let theta = std::f64::consts::PI * hamming_dist as f64 / num_bits as f64;
        theta.cos()
    }
}

/// MinHash for Jaccard similarity on sets.
///
/// Computes k independent hash functions and takes the minimum hash value for each.
/// Probability of equal MinHash = Jaccard similarity.
#[derive(Clone)]
pub struct MinHash {
    /// Hash function parameters: (a, b) for h(x) = (a*x + b) mod p.
    hash_params: Vec<(u64, u64)>,
    /// Large prime for modular arithmetic.
    prime: u64,
    /// Number of hash functions.
    num_hashes: usize,
}

impl MinHash {
    /// Create a new MinHash with k hash functions.
    pub fn new(num_hashes: usize, seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };

        // Mersenne prime for modular hashing
        let prime = (1u64 << 61) - 1;

        let hash_params: Vec<(u64, u64)> = (0..num_hashes)
            .map(|_| {
                let a = rng.gen_range(1..prime);
                let b = rng.gen_range(0..prime);
                (a, b)
            })
            .collect();

        Self {
            hash_params,
            prime,
            num_hashes,
        }
    }

    /// Compute MinHash signature for a set of elements.
    pub fn hash(&self, elements: &[u32]) -> Vec<u64> {
        let mut signature = vec![u64::MAX; self.num_hashes];

        for &elem in elements {
            let elem = elem as u64;
            for (i, &(a, b)) in self.hash_params.iter().enumerate() {
                let h = (a.wrapping_mul(elem).wrapping_add(b)) % self.prime;
                signature[i] = signature[i].min(h);
            }
        }

        signature
    }

    /// Compute MinHash signature for an SDR.
    pub fn hash_sdr(&self, sdr: &Sdr) -> Vec<u64> {
        let positions: Vec<u32> = sdr.to_positions();
        self.hash(&positions)
    }

    /// Estimate Jaccard similarity from MinHash signatures.
    pub fn estimate_jaccard(sig1: &[u64], sig2: &[u64]) -> f64 {
        assert_eq!(sig1.len(), sig2.len());
        let matches = sig1.iter().zip(sig2.iter()).filter(|(&a, &b)| a == b).count();
        matches as f64 / sig1.len() as f64
    }
}

/// BitSampling LSH for binary vectors (SDRs).
///
/// Simply samples random bit positions. If two vectors have similar bits
/// at the sampled positions, they're likely similar overall.
/// Extremely fast for sparse binary vectors.
#[derive(Clone)]
#[allow(dead_code)]
pub struct BitSamplingLsh {
    /// Positions to sample.
    sample_positions: Vec<u32>,
    /// Grid size (stored for potential future use in serialization/debugging).
    grid_size: u32,
}

impl BitSamplingLsh {
    /// Create a new BitSampling LSH.
    pub fn new(grid_size: u32, num_samples: usize, seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };

        let sample_positions: Vec<u32> = (0..num_samples)
            .map(|_| rng.gen_range(0..grid_size))
            .collect();

        Self {
            sample_positions,
            grid_size,
        }
    }

    /// Compute hash signature for an SDR.
    pub fn hash(&self, sdr: &Sdr) -> u64 {
        let mut signature = 0u64;

        for (i, &pos) in self.sample_positions.iter().enumerate().take(64) {
            if sdr.contains(pos) {
                signature |= 1u64 << i;
            }
        }

        signature
    }

    /// Compute extended signature (arbitrary length).
    pub fn hash_extended(&self, sdr: &Sdr) -> Vec<u64> {
        let num_words = (self.sample_positions.len() + 63) / 64;
        let mut signature = vec![0u64; num_words];

        for (i, &pos) in self.sample_positions.iter().enumerate() {
            if sdr.contains(pos) {
                let word = i / 64;
                let bit = i % 64;
                signature[word] |= 1u64 << bit;
            }
        }

        signature
    }
}

/// Multi-table LSH index for approximate nearest neighbor search.
///
/// Uses multiple hash tables with different hash functions to improve recall.
/// Supports banding (concatenating hash values) for precision/recall tradeoff.
pub struct LshIndex<H: LshHasher> {
    /// Hash tables: tables[t] = HashMap<hash_value, vec<data_indices>>.
    tables: Vec<HashMap<u64, Vec<u32>>>,
    /// The hasher for each table.
    hashers: Vec<H>,
    /// Number of tables.
    num_tables: usize,
    /// Number of bands (hash values concatenated per table).
    bands: usize,
}

/// Trait for LSH hash functions.
pub trait LshHasher: Clone {
    /// Compute hash value for a given band.
    fn hash_band(&self, data: &Sdr, band: usize) -> u64;
}

impl LshHasher for MinHash {
    fn hash_band(&self, data: &Sdr, band: usize) -> u64 {
        let sig = self.hash_sdr(data);
        let hashes_per_band = (sig.len() + 15) / 16; // ~16 bands
        let start = band * hashes_per_band;
        let end = (start + hashes_per_band).min(sig.len());

        // Combine hash values in this band
        let mut combined = 0u64;
        for (i, &h) in sig[start..end].iter().enumerate() {
            combined ^= h.rotate_left((i * 7) as u32);
        }
        combined
    }
}

impl LshHasher for BitSamplingLsh {
    fn hash_band(&self, data: &Sdr, band: usize) -> u64 {
        // Each band uses different bit positions
        let sig = self.hash_extended(data);
        sig.get(band).copied().unwrap_or(0)
    }
}

impl<H: LshHasher> LshIndex<H> {
    /// Create a new LSH index.
    pub fn new(hashers: Vec<H>, num_tables: usize, bands: usize) -> Self {
        assert_eq!(hashers.len(), num_tables);

        Self {
            tables: vec![HashMap::new(); num_tables],
            hashers,
            num_tables,
            bands,
        }
    }

    /// Insert a fingerprint into the index.
    pub fn insert(&mut self, idx: u32, data: &Sdr) {
        for (t, hasher) in self.hashers.iter().enumerate() {
            for b in 0..self.bands {
                let hash = hasher.hash_band(data, b);
                self.tables[t].entry(hash).or_default().push(idx);
            }
        }
    }

    /// Query for candidate neighbors.
    /// Returns indices that hash to the same bucket as the query in any table.
    pub fn query(&self, data: &Sdr) -> HashSet<u32> {
        let mut candidates = HashSet::new();

        for (t, hasher) in self.hashers.iter().enumerate() {
            for b in 0..self.bands {
                let hash = hasher.hash_band(data, b);
                if let Some(bucket) = self.tables[t].get(&hash) {
                    candidates.extend(bucket.iter().copied());
                }
            }
        }

        candidates
    }

    /// Query with multi-probe for improved recall.
    /// Probes neighboring buckets in addition to the exact match.
    pub fn query_multiprobe(&self, data: &Sdr, num_probes: usize) -> HashSet<u32> {
        let mut candidates = HashSet::new();

        for (t, hasher) in self.hashers.iter().enumerate() {
            for b in 0..self.bands {
                let hash = hasher.hash_band(data, b);

                // Exact match
                if let Some(bucket) = self.tables[t].get(&hash) {
                    candidates.extend(bucket.iter().copied());
                }

                // Probe neighboring buckets (flip single bits)
                for probe in 0..num_probes.min(64) {
                    let neighbor_hash = hash ^ (1u64 << probe);
                    if let Some(bucket) = self.tables[t].get(&neighbor_hash) {
                        candidates.extend(bucket.iter().copied());
                    }
                }
            }
        }

        candidates
    }

    /// Get statistics about the index.
    pub fn stats(&self) -> LshStats {
        let mut total_buckets = 0;
        let mut total_items = 0;
        let mut max_bucket_size = 0;

        for table in &self.tables {
            total_buckets += table.len();
            for bucket in table.values() {
                total_items += bucket.len();
                max_bucket_size = max_bucket_size.max(bucket.len());
            }
        }

        LshStats {
            num_tables: self.num_tables,
            bands: self.bands,
            total_buckets,
            total_items,
            avg_bucket_size: if total_buckets > 0 {
                total_items as f64 / total_buckets as f64
            } else {
                0.0
            },
            max_bucket_size,
        }
    }
}

/// Statistics about an LSH index.
#[derive(Debug, Clone)]
pub struct LshStats {
    /// Number of hash tables.
    pub num_tables: usize,
    /// Number of bands per table.
    pub bands: usize,
    /// Total number of non-empty buckets.
    pub total_buckets: usize,
    /// Total number of items across all buckets.
    pub total_items: usize,
    /// Average bucket size.
    pub avg_bucket_size: f64,
    /// Maximum bucket size.
    pub max_bucket_size: usize,
}

/// Convenience function to create an LSH index for SDRs using MinHash.
pub fn create_minhash_index(num_hashes: usize, num_tables: usize, seed: Option<u64>) -> LshIndex<MinHash> {
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    let hashers: Vec<MinHash> = (0..num_tables)
        .map(|_| MinHash::new(num_hashes, Some(rng.gen())))
        .collect();

    LshIndex::new(hashers, num_tables, 1)
}

/// Convenience function to create a BitSampling LSH index for SDRs.
pub fn create_bitsampling_index(
    grid_size: u32,
    num_samples: usize,
    num_tables: usize,
    seed: Option<u64>,
) -> LshIndex<BitSamplingLsh> {
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    let hashers: Vec<BitSamplingLsh> = (0..num_tables)
        .map(|_| BitSamplingLsh::new(grid_size, num_samples, Some(rng.gen())))
        .collect();

    LshIndex::new(hashers, num_tables, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_sdrs(n: usize, grid_size: u32) -> Vec<Sdr> {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        (0..n)
            .map(|_| {
                let num_bits = rng.gen_range(20..80);
                let positions: Vec<u32> = (0..num_bits)
                    .map(|_| rng.gen_range(0..grid_size))
                    .collect();
                Sdr::from_positions(&positions, grid_size)
            })
            .collect()
    }

    #[test]
    fn test_simhash() {
        let simhash = SimHash::new(100, 64, Some(42));

        let v1 = vec![1.0; 100];
        let v2 = vec![1.0; 100];
        let v3: Vec<f32> = (0..100).map(|i| if i < 50 { 1.0 } else { -1.0 }).collect();

        let h1 = simhash.hash(&v1);
        let h2 = simhash.hash(&v2);
        let h3 = simhash.hash(&v3);

        // Identical vectors should have identical hashes
        assert_eq!(h1, h2);

        // Different vectors should have different hashes
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_minhash() {
        let minhash = MinHash::new(128, Some(42));

        let set1: Vec<u32> = (0..50).collect();
        let set2: Vec<u32> = (25..75).collect(); // 50% overlap with set1
        let set3: Vec<u32> = (100..150).collect(); // No overlap

        let sig1 = minhash.hash(&set1);
        let sig2 = minhash.hash(&set2);
        let sig3 = minhash.hash(&set3);

        let jaccard_12 = MinHash::estimate_jaccard(&sig1, &sig2);
        let jaccard_13 = MinHash::estimate_jaccard(&sig1, &sig3);

        // set1 and set2 have ~33% Jaccard similarity (25/75)
        assert!(jaccard_12 > 0.2 && jaccard_12 < 0.5);

        // set1 and set3 should have ~0% similarity
        assert!(jaccard_13 < 0.1);
    }

    #[test]
    fn test_bitsampling() {
        let lsh = BitSamplingLsh::new(1024, 64, Some(42));

        // Use denser SDRs to ensure hash collision is unlikely
        let positions1: Vec<u32> = (0..100).collect();
        let positions2: Vec<u32> = (500..600).collect();

        let sdr1 = Sdr::from_positions(&positions1, 1024);
        let sdr2 = Sdr::from_positions(&positions1, 1024); // Identical to sdr1
        let sdr3 = Sdr::from_positions(&positions2, 1024); // Different from sdr1

        let h1 = lsh.hash(&sdr1);
        let h2 = lsh.hash(&sdr2);
        let h3 = lsh.hash(&sdr3);

        // Identical SDRs should have identical hashes
        assert_eq!(h1, h2);

        // Different SDRs with non-overlapping bits should have different hashes
        // With 100 bits set out of 1024 and 64 random samples, ~6 samples should hit,
        // making collision extremely unlikely
        assert_ne!(h1, h3, "Non-overlapping SDRs should have different hashes");
    }

    #[test]
    fn test_lsh_index() {
        let sdrs = create_test_sdrs(100, 1024);
        let mut index = create_minhash_index(64, 4, Some(42));

        // Insert all SDRs
        for (i, sdr) in sdrs.iter().enumerate() {
            index.insert(i as u32, sdr);
        }

        // Query should find candidates
        let candidates = index.query(&sdrs[0]);
        assert!(!candidates.is_empty());
        assert!(candidates.contains(&0)); // Should find itself

        let stats = index.stats();
        assert!(stats.total_buckets > 0);
    }
}
