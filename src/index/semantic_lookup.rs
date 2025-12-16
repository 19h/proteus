//! Sophisticated semantic position lookup engine.
//!
//! This module implements a state-of-the-art system for finding representative
//! words at given positions on the SOM grid, incorporating:
//!
//! 1. **Toroidal topology**: Proper handling of the wrapped SOM manifold
//! 2. **Adaptive kernels**: Density-aware neighborhood expansion
//! 3. **Information-theoretic weighting**: Position importance based on discriminability
//! 4. **Hierarchical clustering**: Multi-scale semantic regions
//! 5. **Kernel convolution**: Soft similarity matching (Cortical.io style)
//!
//! # Theory
//!
//! The SOM organizes words on a 2D manifold where semantic similarity correlates
//! with spatial proximity. A naive position lookup returns only exact matches,
//! but semantics are continuous—we need to capture the "semantic field" around
//! each position.
//!
//! Key insight: Not all positions are equally informative. A position activated
//! by 10,000 words is less discriminative than one activated by 10 words. We
//! use information-theoretic weighting (analogous to IDF in text retrieval) to
//! prioritize rare, specific positions.

use crate::index::topology::ToroidalGrid;
use crate::index::InvertedIndex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Configuration for the semantic lookup engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticLookupConfig {
    /// Base radius for neighborhood expansion (Cortical.io uses 1.8).
    pub base_radius: f64,

    /// Kernel size for convolution (Cortical.io uses 7).
    pub kernel_size: usize,

    /// Sigma for Gaussian kernel (controls decay rate).
    pub kernel_sigma: f64,

    /// Whether to use adaptive radius based on local density.
    pub adaptive_radius: bool,

    /// Minimum radius when using adaptive expansion.
    pub min_radius: f64,

    /// Maximum radius when using adaptive expansion.
    pub max_radius: f64,

    /// Number of hierarchical clustering levels.
    pub cluster_levels: usize,

    /// Target clusters per level (geometric progression).
    pub clusters_per_level: Vec<usize>,

    /// Smoothing parameter for IDF computation (prevents division by zero).
    pub idf_smoothing: f64,
}

impl Default for SemanticLookupConfig {
    fn default() -> Self {
        Self {
            base_radius: 1.8,
            kernel_size: 7,
            kernel_sigma: 1.5,
            adaptive_radius: true,
            min_radius: 1.0,
            max_radius: 4.0,
            cluster_levels: 3,
            clusters_per_level: vec![256, 64, 16], // Fine → Coarse
            idf_smoothing: 1.0,
        }
    }
}

/// Statistics about a position's information content.
#[derive(Debug, Clone)]
pub struct PositionStats {
    /// Number of words that activate this position.
    pub word_count: usize,

    /// Information content: -log(P(position)) = log(total_words / position_words).
    pub information_content: f64,

    /// Inverse document frequency analog: log(N / (1 + df)).
    pub idf: f64,

    /// Local density: average words per position in neighborhood.
    pub local_density: f64,

    /// Cluster membership at each hierarchical level.
    pub cluster_ids: Vec<usize>,
}

/// A semantic cluster of positions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCluster {
    /// Cluster ID.
    pub id: usize,

    /// Hierarchical level (0 = finest, higher = coarser).
    pub level: usize,

    /// Centroid position (linear index).
    pub centroid: u32,

    /// Member positions.
    pub members: Vec<u32>,

    /// Representative words for this cluster.
    pub representative_words: Vec<String>,

    /// Average IDF of member positions.
    pub avg_idf: f64,

    /// Radius (max distance from centroid to any member).
    pub radius: f64,
}

/// Result of a semantic lookup query.
#[derive(Debug, Clone)]
pub struct LookupResult {
    /// Word and its relevance score.
    pub word: String,

    /// Overall relevance score (combines multiple factors).
    pub score: f64,

    /// Distance to nearest query position.
    pub min_distance: f64,

    /// Number of query positions this word overlaps.
    pub overlap_count: u32,

    /// Weighted overlap (accounts for position importance).
    pub weighted_overlap: f64,

    /// Breakdown of contributing factors.
    pub factors: LookupFactors,
}

/// Breakdown of factors contributing to a lookup score.
#[derive(Debug, Clone, Default)]
pub struct LookupFactors {
    /// Spatial proximity contribution.
    pub spatial_score: f64,

    /// Information content contribution.
    pub information_score: f64,

    /// Cluster coherence contribution.
    pub cluster_score: f64,

    /// Kernel-weighted similarity.
    pub kernel_score: f64,
}

/// The semantic lookup engine.
pub struct SemanticLookupEngine {
    /// Configuration.
    config: SemanticLookupConfig,

    /// Toroidal grid operations.
    grid: ToroidalGrid,

    /// Pre-computed position statistics.
    position_stats: Vec<PositionStats>,

    /// Hierarchical clusters at each level.
    clusters: Vec<Vec<SemanticCluster>>,

    /// Position to cluster mapping at each level.
    position_to_cluster: Vec<Vec<usize>>,

    /// Pre-computed Gaussian kernel.
    kernel: Vec<Vec<f64>>,

    /// Total vocabulary size (for IDF computation).
    vocab_size: usize,

    /// Whether the engine has been initialized.
    initialized: bool,
}

impl SemanticLookupEngine {
    /// Create a new semantic lookup engine.
    pub fn new(dimension: u32, config: SemanticLookupConfig) -> Self {
        let grid = ToroidalGrid::new(dimension);
        let kernel = Self::create_gaussian_kernel(config.kernel_size, config.kernel_sigma);

        Self {
            config,
            grid,
            position_stats: Vec::new(),
            clusters: Vec::new(),
            position_to_cluster: Vec::new(),
            kernel,
            vocab_size: 0,
            initialized: false,
        }
    }

    /// Create with default configuration.
    pub fn with_dimension(dimension: u32) -> Self {
        Self::new(dimension, SemanticLookupConfig::default())
    }

    /// Initialize the engine from an inverted index.
    ///
    /// This pre-computes:
    /// 1. Position statistics (word counts, IDF, density)
    /// 2. Hierarchical clusters
    /// 3. Representative words per cluster
    pub fn initialize(&mut self, index: &InvertedIndex) {
        self.vocab_size = index.len();

        eprintln!("[semantic_lookup] Initializing engine...");
        eprintln!("[semantic_lookup]   Grid: {}x{}", self.grid.dimension(), self.grid.dimension());
        eprintln!("[semantic_lookup]   Vocabulary: {} words", self.vocab_size);

        // Step 1: Compute position statistics
        eprintln!("[semantic_lookup] Step 1: Computing position statistics...");
        self.compute_position_stats(index);

        // Step 2: Build hierarchical clusters
        eprintln!("[semantic_lookup] Step 2: Building hierarchical clusters...");
        self.build_hierarchical_clusters(index);

        // Step 3: Find representative words for clusters
        eprintln!("[semantic_lookup] Step 3: Computing cluster representatives...");
        self.compute_cluster_representatives(index);

        self.initialized = true;
        eprintln!("[semantic_lookup] Initialization complete.");
    }

    /// Check if the engine is initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Compute statistics for each position.
    fn compute_position_stats(&mut self, index: &InvertedIndex) {
        let grid_size = self.grid.grid_size() as usize;
        let vocab_size = self.vocab_size as f64;

        self.position_stats = (0..grid_size)
            .into_par_iter()
            .map(|pos| {
                let words = index.words_at_position(pos as u32);
                let word_count = words.len();

                // Information content: -log(P(position is active))
                let p = (word_count as f64 + self.config.idf_smoothing) / (vocab_size + self.config.idf_smoothing);
                let information_content = -p.ln();

                // IDF: log(N / (1 + df))
                let idf = (vocab_size / (1.0 + word_count as f64)).ln();

                // Local density: average word count in neighborhood
                let neighbors = self.grid.neighborhood(pos as u32, self.config.base_radius);
                let neighbor_counts: Vec<usize> = neighbors
                    .iter()
                    .map(|&n| index.words_at_position(n).len())
                    .collect();
                let local_density = if neighbor_counts.is_empty() {
                    word_count as f64
                } else {
                    neighbor_counts.iter().sum::<usize>() as f64 / neighbor_counts.len() as f64
                };

                PositionStats {
                    word_count,
                    information_content,
                    idf,
                    local_density,
                    cluster_ids: Vec::new(), // Filled in later
                }
            })
            .collect();
    }

    /// Build hierarchical clusters using k-means on the position space.
    fn build_hierarchical_clusters(&mut self, index: &InvertedIndex) {
        let grid_size = self.grid.grid_size();

        // Get positions with at least one word (active positions)
        let active_positions: Vec<u32> = (0..grid_size)
            .filter(|&p| self.position_stats[p as usize].word_count > 0)
            .collect();

        eprintln!("[semantic_lookup]   Active positions: {}", active_positions.len());

        self.clusters = Vec::new();
        self.position_to_cluster = vec![vec![0; grid_size as usize]; self.config.cluster_levels];

        for level in 0..self.config.cluster_levels {
            let k = self.config.clusters_per_level.get(level).copied().unwrap_or(16);
            eprintln!("[semantic_lookup]   Level {}: {} clusters", level, k);

            let level_clusters = self.kmeans_cluster(&active_positions, k, level, index);

            // Update position-to-cluster mapping
            for cluster in &level_clusters {
                for &pos in &cluster.members {
                    self.position_to_cluster[level][pos as usize] = cluster.id;
                }
            }

            self.clusters.push(level_clusters);
        }

        // Update position stats with cluster IDs
        for pos in 0..grid_size as usize {
            self.position_stats[pos].cluster_ids = (0..self.config.cluster_levels)
                .map(|level| self.position_to_cluster[level][pos])
                .collect();
        }
    }

    /// K-means clustering on the toroidal grid.
    fn kmeans_cluster(
        &self,
        positions: &[u32],
        k: usize,
        level: usize,
        _index: &InvertedIndex,
    ) -> Vec<SemanticCluster> {
        if positions.is_empty() || k == 0 {
            return Vec::new();
        }

        let k = k.min(positions.len());
        let max_iterations = 50;

        // Initialize centroids using k-means++ strategy
        let mut centroids = self.kmeans_plus_plus_init(positions, k);
        let mut assignments = vec![0usize; positions.len()];

        for _iteration in 0..max_iterations {
            let mut changed = false;

            // Assign each position to nearest centroid
            for (i, &pos) in positions.iter().enumerate() {
                let nearest = centroids
                    .iter()
                    .enumerate()
                    .map(|(c_idx, &centroid)| (c_idx, self.grid.wrapped_distance(pos, centroid)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                if assignments[i] != nearest {
                    assignments[i] = nearest;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            for c_idx in 0..k {
                let members: Vec<u32> = positions
                    .iter()
                    .zip(assignments.iter())
                    .filter(|(_, &a)| a == c_idx)
                    .map(|(&p, _)| p)
                    .collect();

                if !members.is_empty() {
                    centroids[c_idx] = self.compute_centroid(&members);
                }
            }
        }

        // Build cluster objects
        let mut clusters = Vec::new();
        for c_idx in 0..k {
            let members: Vec<u32> = positions
                .iter()
                .zip(assignments.iter())
                .filter(|(_, &a)| a == c_idx)
                .map(|(&p, _)| p)
                .collect();

            if members.is_empty() {
                continue;
            }

            let centroid = centroids[c_idx];

            // Compute radius
            let radius = members
                .iter()
                .map(|&m| self.grid.wrapped_distance(centroid, m))
                .fold(0.0_f64, |a, b| a.max(b));

            // Compute average IDF
            let avg_idf = members
                .iter()
                .map(|&m| self.position_stats[m as usize].idf)
                .sum::<f64>() / members.len() as f64;

            clusters.push(SemanticCluster {
                id: c_idx,
                level,
                centroid,
                members,
                representative_words: Vec::new(), // Filled in later
                avg_idf,
                radius,
            });
        }

        clusters
    }

    /// K-means++ initialization for better starting centroids.
    fn kmeans_plus_plus_init(&self, positions: &[u32], k: usize) -> Vec<u32> {
        use rand::seq::SliceRandom;
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let mut centroids = Vec::with_capacity(k);

        // First centroid: random
        if let Some(&first) = positions.choose(&mut rng) {
            centroids.push(first);
        } else {
            return centroids;
        }

        // Remaining centroids: probability proportional to squared distance
        while centroids.len() < k {
            let distances: Vec<f64> = positions
                .iter()
                .map(|&p| {
                    centroids
                        .iter()
                        .map(|&c| self.grid.wrapped_distance(p, c))
                        .fold(f64::MAX, |a, b| a.min(b))
                        .powi(2)
                })
                .collect();

            let total: f64 = distances.iter().sum();
            if total == 0.0 {
                break;
            }

            let threshold = rng.gen::<f64>() * total;
            let mut cumsum = 0.0;

            for (i, &d) in distances.iter().enumerate() {
                cumsum += d;
                if cumsum >= threshold {
                    centroids.push(positions[i]);
                    break;
                }
            }
        }

        centroids
    }

    /// Compute centroid of a set of positions on the toroidal grid.
    ///
    /// This uses circular mean to properly handle wrapping.
    fn compute_centroid(&self, positions: &[u32]) -> u32 {
        if positions.is_empty() {
            return 0;
        }

        let dim = self.grid.dimension() as f64;

        // Convert to angles and use circular mean
        let (sin_row, cos_row, sin_col, cos_col) = positions.iter().fold(
            (0.0, 0.0, 0.0, 0.0),
            |(sr, cr, sc, cc), &p| {
                let pos = self.grid.to_2d(p);
                let angle_row = 2.0 * std::f64::consts::PI * pos.row as f64 / dim;
                let angle_col = 2.0 * std::f64::consts::PI * pos.col as f64 / dim;
                (
                    sr + angle_row.sin(),
                    cr + angle_row.cos(),
                    sc + angle_col.sin(),
                    cc + angle_col.cos(),
                )
            },
        );

        let n = positions.len() as f64;
        let mean_angle_row = (sin_row / n).atan2(cos_row / n);
        let mean_angle_col = (sin_col / n).atan2(cos_col / n);

        // Convert back to grid coordinates
        let row = ((mean_angle_row / (2.0 * std::f64::consts::PI) * dim + dim) % dim) as u32;
        let col = ((mean_angle_col / (2.0 * std::f64::consts::PI) * dim + dim) % dim) as u32;

        row * self.grid.dimension() + col
    }

    /// Find representative words for each cluster.
    fn compute_cluster_representatives(&mut self, index: &InvertedIndex) {
        for level_clusters in &mut self.clusters {
            for cluster in level_clusters {
                // Get all words from cluster member positions
                let mut word_scores: HashMap<String, f64> = HashMap::new();

                for &pos in &cluster.members {
                    let pos_idf = self.position_stats[pos as usize].idf;

                    for word in index.words_at_position(pos) {
                        *word_scores.entry(word).or_insert(0.0) += pos_idf;
                    }
                }

                // Sort by score and take top representatives
                let mut scored: Vec<(String, f64)> = word_scores.into_iter().collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                cluster.representative_words = scored
                    .into_iter()
                    .take(10)
                    .map(|(w, _)| w)
                    .collect();
            }
        }
    }

    /// Create a Gaussian kernel for convolution.
    fn create_gaussian_kernel(size: usize, sigma: f64) -> Vec<Vec<f64>> {
        let center = (size / 2) as f64;
        let mut kernel = vec![vec![0.0; size]; size];
        let mut sum = 0.0;

        for i in 0..size {
            for j in 0..size {
                let di = i as f64 - center;
                let dj = j as f64 - center;
                let dist_sq = di * di + dj * dj;
                let value = (-dist_sq / (2.0 * sigma * sigma)).exp();
                kernel[i][j] = value;
                sum += value;
            }
        }

        // Normalize
        for row in &mut kernel {
            for val in row {
                *val /= sum;
            }
        }

        kernel
    }

    /// Compute adaptive radius based on local density.
    fn adaptive_radius(&self, position: u32) -> f64 {
        if !self.config.adaptive_radius {
            return self.config.base_radius;
        }

        let stats = &self.position_stats[position as usize];
        let avg_density = self.position_stats.iter()
            .map(|s| s.local_density)
            .sum::<f64>() / self.position_stats.len() as f64;

        // Inverse relationship: denser regions → smaller radius
        let density_ratio = avg_density / (stats.local_density + 1.0);
        let radius = self.config.base_radius * density_ratio.sqrt();

        radius.clamp(self.config.min_radius, self.config.max_radius)
    }

    /// Main lookup function: find representative words for given positions.
    ///
    /// This implements a sophisticated multi-factor scoring:
    /// 1. Spatial proximity (kernel-weighted distance)
    /// 2. Information content (IDF weighting)
    /// 3. Cluster coherence (same cluster = bonus)
    /// 4. Multi-position overlap (words covering multiple query positions)
    pub fn lookup(
        &self,
        positions: &[u32],
        index: &InvertedIndex,
        k: usize,
    ) -> Vec<LookupResult> {
        if !self.initialized || positions.is_empty() {
            return Vec::new();
        }

        // Expand each position to its neighborhood with weights
        let mut expanded_positions: HashMap<u32, f64> = HashMap::new();

        for &pos in positions {
            let radius = self.adaptive_radius(pos);
            let neighbors = self.grid.neighborhood_weighted(pos, radius);

            for (neighbor, distance) in neighbors {
                let weight = self.grid.gaussian_weight(distance, self.config.kernel_sigma);
                let idf = self.position_stats[neighbor as usize].idf;

                // Combined weight: spatial × information content
                let combined = weight * (1.0 + idf);

                *expanded_positions.entry(neighbor).or_insert(0.0) += combined;
            }
        }

        // Score each word based on expanded position overlap
        let mut word_scores: HashMap<String, (f64, f64, u32, LookupFactors)> = HashMap::new();

        for (&pos, &weight) in &expanded_positions {
            let pos_stats = &self.position_stats[pos as usize];

            for word in index.words_at_position(pos) {
                let entry = word_scores.entry(word.clone()).or_insert_with(|| {
                    (0.0, f64::MAX, 0, LookupFactors::default())
                });

                // Accumulate weighted overlap
                entry.0 += weight;

                // Track minimum distance to any query position
                let min_dist = positions.iter()
                    .map(|&q| self.grid.wrapped_distance(q, pos))
                    .fold(f64::MAX, |a, b| a.min(b));
                entry.1 = entry.1.min(min_dist);

                // Count overlapping query positions
                entry.2 += 1;

                // Accumulate factor breakdowns
                entry.3.spatial_score += self.grid.gaussian_weight(min_dist, self.config.kernel_sigma);
                entry.3.information_score += pos_stats.idf;
                entry.3.kernel_score += weight;
            }
        }

        // Add cluster coherence bonus
        let query_clusters: HashSet<usize> = positions.iter()
            .flat_map(|&p| self.position_stats[p as usize].cluster_ids.iter().cloned())
            .collect();

        for (word, (_, _, _, ref mut factors)) in &mut word_scores {
            // Check if word appears in same clusters as query
            if let Some(fp) = self.get_word_fingerprint_positions(word, index) {
                let word_clusters: HashSet<usize> = fp.iter()
                    .flat_map(|&p| self.position_stats[p as usize].cluster_ids.iter().cloned())
                    .collect();

                let cluster_overlap = query_clusters.intersection(&word_clusters).count();
                factors.cluster_score = cluster_overlap as f64 / query_clusters.len().max(1) as f64;
            }
        }

        // Compute final scores and build results
        let mut results: Vec<LookupResult> = word_scores
            .into_iter()
            .map(|(word, (weighted_overlap, min_dist, overlap_count, factors))| {
                // Combined score formula
                let score = weighted_overlap
                    * (1.0 + factors.cluster_score)
                    * (1.0 + (overlap_count as f64).ln());

                LookupResult {
                    word,
                    score,
                    min_distance: min_dist,
                    overlap_count,
                    weighted_overlap,
                    factors,
                }
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(k);

        results
    }

    /// Lookup with single position (convenience method).
    pub fn lookup_single(
        &self,
        position: u32,
        index: &InvertedIndex,
        k: usize,
    ) -> Vec<LookupResult> {
        self.lookup(&[position], index, k)
    }

    /// Get cluster information for a position.
    pub fn get_cluster_info(&self, position: u32, level: usize) -> Option<&SemanticCluster> {
        if level >= self.clusters.len() {
            return None;
        }

        let cluster_id = self.position_to_cluster.get(level)?
            .get(position as usize)?;

        self.clusters.get(level)?
            .iter()
            .find(|c| c.id == *cluster_id)
    }

    /// Get all clusters at a given level.
    pub fn get_clusters(&self, level: usize) -> Option<&Vec<SemanticCluster>> {
        self.clusters.get(level)
    }

    /// Get position statistics.
    pub fn get_position_stats(&self, position: u32) -> Option<&PositionStats> {
        self.position_stats.get(position as usize)
    }

    /// Helper to get word fingerprint positions from index.
    fn get_word_fingerprint_positions(&self, _word: &str, _index: &InvertedIndex) -> Option<Vec<u32>> {
        // This would need access to the forward index (word → positions)
        // For now, return None and skip cluster coherence for unknown words
        None
    }

    /// Apply kernel convolution to a fingerprint (Cortical.io style).
    ///
    /// Transforms a binary fingerprint into a continuous "heat map" where
    /// each active position becomes a Gaussian bump.
    pub fn convolve_fingerprint(&self, positions: &[u32]) -> Vec<f64> {
        let grid_size = self.grid.grid_size() as usize;
        let dim = self.grid.dimension() as usize;
        let k_half = self.config.kernel_size / 2;

        let mut result = vec![0.0; grid_size];

        for &pos in positions {
            let pos_2d = self.grid.to_2d(pos);
            let row = pos_2d.row as i32;
            let col = pos_2d.col as i32;

            // Apply kernel around this position (with wrapping)
            for ki in 0..self.config.kernel_size {
                for kj in 0..self.config.kernel_size {
                    let di = ki as i32 - k_half as i32;
                    let dj = kj as i32 - k_half as i32;

                    // Wrap coordinates
                    let nr = ((row + di) % dim as i32 + dim as i32) % dim as i32;
                    let nc = ((col + dj) % dim as i32 + dim as i32) % dim as i32;

                    let target = (nr as usize) * dim + (nc as usize);
                    result[target] += self.kernel[ki][kj];
                }
            }
        }

        result
    }

    /// Compute weighted similarity between two fingerprints using kernel convolution.
    ///
    /// This is the sophisticated similarity metric used by Cortical.io.
    pub fn weighted_similarity(&self, fp1: &[u32], fp2: &[u32]) -> f64 {
        let conv1 = self.convolve_fingerprint(fp1);
        let conv2 = self.convolve_fingerprint(fp2);

        // Dot product of convolved representations
        let dot: f64 = conv1.iter().zip(conv2.iter()).map(|(a, b)| a * b).sum();

        // Normalize
        let norm1: f64 = conv1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = conv2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot / (norm1 * norm2)
        }
    }

    /// Get interpretable explanation for why a word was returned.
    pub fn explain_result(&self, result: &LookupResult, _query_positions: &[u32]) -> String {
        let mut explanation = format!("Word: '{}'\n", result.word);
        explanation += &format!("  Overall score: {:.4}\n", result.score);
        explanation += &format!("  Min distance to query: {:.2}\n", result.min_distance);
        explanation += &format!("  Query positions covered: {}\n", result.overlap_count);
        explanation += &format!("  Weighted overlap: {:.4}\n", result.weighted_overlap);
        explanation += "\n  Factor breakdown:\n";
        explanation += &format!("    Spatial proximity: {:.4}\n", result.factors.spatial_score);
        explanation += &format!("    Information content: {:.4}\n", result.factors.information_score);
        explanation += &format!("    Cluster coherence: {:.4}\n", result.factors.cluster_score);
        explanation += &format!("    Kernel-weighted: {:.4}\n", result.factors.kernel_score);
        explanation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_kernel() {
        let kernel = SemanticLookupEngine::create_gaussian_kernel(7, 1.5);

        // Center should be highest
        assert!(kernel[3][3] > kernel[0][0]);

        // Should be symmetric
        assert!((kernel[0][0] - kernel[6][6]).abs() < 1e-10);
        assert!((kernel[0][3] - kernel[6][3]).abs() < 1e-10);

        // Should sum to 1 (normalized)
        let sum: f64 = kernel.iter().flat_map(|r| r.iter()).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_circular_centroid() {
        let engine = SemanticLookupEngine::with_dimension(128);

        // Positions near the edge should wrap properly
        let positions = vec![0, 1, 127, 128, 129, 128 * 127, 128 * 128 - 1];
        let centroid = engine.compute_centroid(&positions);

        // Centroid should be near the corner (0,0) due to wrapping
        let pos = engine.grid.to_2d(centroid);
        assert!(pos.row <= 2 || pos.row >= 126);
        assert!(pos.col <= 2 || pos.col >= 126);
    }
}
