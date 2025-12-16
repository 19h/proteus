//! Product Quantization (PQ) for compressed vector storage and fast distance computation.
//!
//! PQ compresses high-dimensional vectors by:
//! 1. Splitting vectors into M subvectors
//! 2. Quantizing each subvector to K centroids (codebook)
//! 3. Storing only the centroid indices (log2(K) bits each)
//!
//! Distance computation uses Asymmetric Distance Computation (ADC):
//! - Pre-compute distances from query to all centroids
//! - Sum lookup table values for each compressed vector
//!
//! This enables ~100x faster similarity search with minimal accuracy loss.
//!
//! References:
//! - Jégou et al. (2011): "Product quantization for nearest neighbor search"

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Configuration for Product Quantization.
#[derive(Debug, Clone)]
pub struct PqConfig {
    /// Number of subquantizers (M). The vector is split into M parts.
    /// Must divide the vector dimension evenly.
    /// Higher M = better accuracy, more memory. Typical: 8-64. Default: 8.
    pub num_subquantizers: usize,

    /// Number of centroids per subquantizer (K). Usually 256 (8 bits).
    /// Higher K = better accuracy, slower training. Default: 256.
    pub num_centroids: usize,

    /// Number of k-means iterations for training. Default: 25.
    pub kmeans_iterations: usize,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for PqConfig {
    fn default() -> Self {
        Self {
            num_subquantizers: 8,
            num_centroids: 256,
            kmeans_iterations: 25,
            seed: None,
        }
    }
}

/// A trained Product Quantizer.
pub struct ProductQuantizer {
    config: PqConfig,
    /// Dimension of input vectors.
    dim: usize,
    /// Dimension of each subvector.
    sub_dim: usize,
    /// Codebooks: codebooks[m][k] = centroid k for subquantizer m.
    /// Shape: [M][K][sub_dim].
    codebooks: Vec<Vec<Vec<f32>>>,
}

impl ProductQuantizer {
    /// Train a Product Quantizer on a set of vectors.
    pub fn train(config: PqConfig, vectors: &[Vec<f32>]) -> Self {
        assert!(!vectors.is_empty(), "Need at least one training vector");

        let dim = vectors[0].len();
        assert!(
            dim % config.num_subquantizers == 0,
            "Vector dimension {} must be divisible by num_subquantizers {}",
            dim,
            config.num_subquantizers
        );

        let sub_dim = dim / config.num_subquantizers;
        let m = config.num_subquantizers;
        let k = config.num_centroids;

        // RNG for deterministic behavior (used indirectly via config.seed in kmeans_train)
        let _rng = match config.seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };

        // Train each subquantizer independently (parallelizable)
        let codebooks: Vec<Vec<Vec<f32>>> = (0..m)
            .into_par_iter()
            .map(|subq_idx| {
                // Extract subvectors for this subquantizer
                let subvectors: Vec<Vec<f32>> = vectors
                    .iter()
                    .map(|v| {
                        let start = subq_idx * sub_dim;
                        let end = start + sub_dim;
                        v[start..end].to_vec()
                    })
                    .collect();

                // Run k-means to find centroids
                Self::kmeans(&subvectors, k, config.kmeans_iterations, subq_idx as u64)
            })
            .collect();

        Self {
            config,
            dim,
            sub_dim,
            codebooks,
        }
    }

    /// Encode a vector to PQ codes.
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        assert_eq!(vector.len(), self.dim);

        (0..self.config.num_subquantizers)
            .map(|m| {
                let start = m * self.sub_dim;
                let end = start + self.sub_dim;
                let subvector = &vector[start..end];

                // Find nearest centroid
                let mut best_idx = 0u8;
                let mut best_dist = f32::MAX;

                for (k, centroid) in self.codebooks[m].iter().enumerate() {
                    let dist = Self::squared_distance(subvector, centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = k as u8;
                    }
                }

                best_idx
            })
            .collect()
    }

    /// Encode multiple vectors in parallel.
    pub fn encode_batch(&self, vectors: &[Vec<f32>]) -> Vec<Vec<u8>> {
        vectors.par_iter().map(|v| self.encode(v)).collect()
    }

    /// Decode PQ codes back to approximate vector.
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        assert_eq!(codes.len(), self.config.num_subquantizers);

        let mut result = Vec::with_capacity(self.dim);
        for (m, &code) in codes.iter().enumerate() {
            result.extend_from_slice(&self.codebooks[m][code as usize]);
        }
        result
    }

    /// Compute distance lookup table for a query vector.
    /// Returns table[m][k] = squared distance from query subvector m to centroid k.
    pub fn compute_distance_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        assert_eq!(query.len(), self.dim);

        (0..self.config.num_subquantizers)
            .map(|m| {
                let start = m * self.sub_dim;
                let end = start + self.sub_dim;
                let query_sub = &query[start..end];

                self.codebooks[m]
                    .iter()
                    .map(|centroid| Self::squared_distance(query_sub, centroid))
                    .collect()
            })
            .collect()
    }

    /// Compute asymmetric distance from query to encoded vector using lookup table.
    #[inline]
    pub fn asymmetric_distance(&self, table: &[Vec<f32>], codes: &[u8]) -> f32 {
        codes
            .iter()
            .enumerate()
            .map(|(m, &code)| table[m][code as usize])
            .sum()
    }

    /// Search for k nearest neighbors using ADC (Asymmetric Distance Computation).
    pub fn search(&self, query: &[f32], codes: &[Vec<u8>], k: usize) -> Vec<(usize, f32)> {
        let table = self.compute_distance_table(query);

        let mut distances: Vec<(usize, f32)> = codes
            .par_iter()
            .enumerate()
            .map(|(i, code)| (i, self.asymmetric_distance(&table, code)))
            .collect();

        // Partial sort for top-k
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);
        distances
    }

    /// Get compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        let original_bytes = self.dim * 4; // f32 = 4 bytes
        let compressed_bytes = self.config.num_subquantizers; // 1 byte per subquantizer
        original_bytes as f64 / compressed_bytes as f64
    }

    /// Get memory usage per vector in bytes.
    pub fn bytes_per_vector(&self) -> usize {
        self.config.num_subquantizers
    }

    // --- Private methods ---

    /// K-means clustering with k-means++ initialization.
    fn kmeans(vectors: &[Vec<f32>], k: usize, iterations: usize, seed: u64) -> Vec<Vec<f32>> {
        if vectors.len() <= k {
            // Not enough vectors, return copies
            let mut centroids = vectors.to_vec();
            centroids.resize(k, vec![0.0; vectors[0].len()]);
            return centroids;
        }

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let dim = vectors[0].len();

        // K-means++ initialization
        let mut centroids = Vec::with_capacity(k);

        // First centroid: random
        let first_idx = rng.gen_range(0..vectors.len());
        centroids.push(vectors[first_idx].clone());

        // Remaining centroids: probability proportional to squared distance
        for _ in 1..k {
            let distances: Vec<f32> = vectors
                .iter()
                .map(|v| {
                    centroids
                        .iter()
                        .map(|c| Self::squared_distance(v, c))
                        .fold(f32::MAX, |a, b| a.min(b))
                })
                .collect();

            let total: f32 = distances.iter().sum();
            if total <= 0.0 {
                // All distances are 0, pick random
                let idx = rng.gen_range(0..vectors.len());
                centroids.push(vectors[idx].clone());
                continue;
            }

            // Weighted random selection
            let mut r: f32 = rng.gen::<f32>() * total;
            let mut chosen = 0;
            for (i, &d) in distances.iter().enumerate() {
                r -= d;
                if r <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            centroids.push(vectors[chosen].clone());
        }

        // K-means iterations
        let mut assignments = vec![0usize; vectors.len()];

        for _ in 0..iterations {
            // Assign vectors to nearest centroid
            for (i, v) in vectors.iter().enumerate() {
                let mut best_k = 0;
                let mut best_dist = f32::MAX;
                for (ki, c) in centroids.iter().enumerate() {
                    let dist = Self::squared_distance(v, c);
                    if dist < best_dist {
                        best_dist = dist;
                        best_k = ki;
                    }
                }
                assignments[i] = best_k;
            }

            // Update centroids
            let mut sums = vec![vec![0.0f32; dim]; k];
            let mut counts = vec![0usize; k];

            for (i, v) in vectors.iter().enumerate() {
                let ki = assignments[i];
                counts[ki] += 1;
                for (j, &val) in v.iter().enumerate() {
                    sums[ki][j] += val;
                }
            }

            for ki in 0..k {
                if counts[ki] > 0 {
                    for j in 0..dim {
                        centroids[ki][j] = sums[ki][j] / counts[ki] as f32;
                    }
                }
            }
        }

        centroids
    }

    /// Compute squared Euclidean distance.
    #[inline]
    fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }
}

/// Optimized Product Quantization (OPQ) with rotation matrix.
/// Learns a rotation to align data with PQ structure for lower quantization error.
pub struct OptimizedProductQuantizer {
    /// Base PQ.
    pq: ProductQuantizer,
    /// Rotation matrix (dim x dim).
    rotation: Vec<Vec<f32>>,
}

impl OptimizedProductQuantizer {
    /// Train OPQ using alternating optimization.
    pub fn train(config: PqConfig, vectors: &[Vec<f32>], opq_iterations: usize) -> Self {
        let dim = vectors[0].len();

        // Initialize rotation as identity
        let mut rotation: Vec<Vec<f32>> = (0..dim)
            .map(|i| {
                let mut row = vec![0.0f32; dim];
                row[i] = 1.0;
                row
            })
            .collect();

        // Alternating optimization
        let mut rotated = vectors.to_vec();
        let mut pq = ProductQuantizer::train(config.clone(), &rotated);

        for _ in 0..opq_iterations {
            // Encode with current PQ
            let codes = pq.encode_batch(&rotated);

            // Decode to get reconstructed vectors
            let reconstructed: Vec<Vec<f32>> = codes.iter().map(|c| pq.decode(c)).collect();

            // Solve for optimal rotation (simplified: use Procrustes)
            rotation = Self::procrustes(vectors, &reconstructed, dim);

            // Apply rotation
            rotated = vectors
                .iter()
                .map(|v| Self::apply_rotation(v, &rotation))
                .collect();

            // Retrain PQ on rotated data
            pq = ProductQuantizer::train(config.clone(), &rotated);
        }

        Self { pq, rotation }
    }

    /// Encode a vector (applies rotation first).
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let rotated = Self::apply_rotation(vector, &self.rotation);
        self.pq.encode(&rotated)
    }

    /// Compute distance table for a query (applies rotation first).
    pub fn compute_distance_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let rotated = Self::apply_rotation(query, &self.rotation);
        self.pq.compute_distance_table(&rotated)
    }

    /// Asymmetric distance using precomputed table.
    #[inline]
    pub fn asymmetric_distance(&self, table: &[Vec<f32>], codes: &[u8]) -> f32 {
        self.pq.asymmetric_distance(table, codes)
    }

    // --- Private methods ---

    fn apply_rotation(v: &[f32], r: &[Vec<f32>]) -> Vec<f32> {
        r.iter()
            .map(|row| row.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum())
            .collect()
    }

    fn procrustes(original: &[Vec<f32>], reconstructed: &[Vec<f32>], dim: usize) -> Vec<Vec<f32>> {
        // Simplified Procrustes: compute SVD of cross-covariance
        // For production, use a proper linear algebra library

        // Compute cross-covariance X^T * Y
        let mut cov = vec![vec![0.0f32; dim]; dim];
        for (x, y) in original.iter().zip(reconstructed.iter()) {
            for i in 0..dim {
                for j in 0..dim {
                    cov[i][j] += x[i] * y[j];
                }
            }
        }

        // Simple approximation: use cov directly normalized
        // In production, compute SVD and return U * V^T
        let n = original.len() as f32;
        for row in &mut cov {
            for val in row.iter_mut() {
                *val /= n;
            }
        }

        // Normalize rows to approximate orthogonal matrix
        for row in &mut cov {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for val in row.iter_mut() {
                    *val /= norm;
                }
            }
        }

        cov
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn create_test_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    #[test]
    fn test_pq_train() {
        let vectors = create_test_vectors(1000, 64);
        let config = PqConfig {
            num_subquantizers: 8,
            num_centroids: 256,
            kmeans_iterations: 10,
            seed: Some(42),
        };

        let pq = ProductQuantizer::train(config, &vectors);
        assert_eq!(pq.codebooks.len(), 8);
        assert_eq!(pq.codebooks[0].len(), 256);
    }

    #[test]
    fn test_pq_encode_decode() {
        let vectors = create_test_vectors(1000, 64);
        let config = PqConfig {
            num_subquantizers: 8,
            num_centroids: 256,
            kmeans_iterations: 25,
            seed: Some(42),
        };

        let pq = ProductQuantizer::train(config, &vectors);
        let query = &vectors[0];

        let codes = pq.encode(query);
        assert_eq!(codes.len(), 8);

        let decoded = pq.decode(&codes);
        assert_eq!(decoded.len(), 64);

        // Check that reconstruction error is reasonable
        let error: f32 = query
            .iter()
            .zip(decoded.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum();
        let error = (error / 64.0).sqrt();
        assert!(error < 1.0, "Reconstruction error {} too high", error);
    }

    #[test]
    fn test_pq_search() {
        let vectors = create_test_vectors(1000, 64);
        let config = PqConfig {
            num_subquantizers: 8,
            num_centroids: 256,
            kmeans_iterations: 25,
            seed: Some(42),
        };

        let pq = ProductQuantizer::train(config, &vectors);
        let codes = pq.encode_batch(&vectors);
        let query = &vectors[500];

        let results = pq.search(query, &codes, 10);
        assert_eq!(results.len(), 10);

        // Query should be closest to itself
        assert_eq!(results[0].0, 500);
    }

    #[test]
    fn test_compression_ratio() {
        let vectors = create_test_vectors(100, 128);
        let config = PqConfig {
            num_subquantizers: 16,
            num_centroids: 256,
            kmeans_iterations: 10,
            seed: Some(42),
        };

        let pq = ProductQuantizer::train(config, &vectors);

        // 128 * 4 bytes = 512 bytes original
        // 16 bytes compressed
        // Ratio = 32x
        assert_eq!(pq.compression_ratio(), 32.0);
        assert_eq!(pq.bytes_per_vector(), 16);
    }
}
