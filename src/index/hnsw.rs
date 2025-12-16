//! Hierarchical Navigable Small World (HNSW) graph index for O(log n) ANN search.
//!
//! HNSW is the state-of-the-art algorithm for approximate nearest neighbor search,
//! offering logarithmic query time with high recall. This implementation is optimized
//! for sparse binary fingerprints (SDRs).
//!
//! Key features:
//! - Hierarchical multi-layer graph structure
//! - Greedy beam search with backtracking
//! - Optimized for Jaccard/overlap similarity on binary vectors
//! - SIMD-accelerated distance computations
//!
//! References:
//! - Malkov & Yashunin (2018): "Efficient and robust approximate nearest neighbor search
//!   using Hierarchical Navigable Small World graphs"

use crate::fingerprint::Sdr;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

/// Configuration for HNSW index construction.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Maximum number of connections per node at layer 0.
    /// Higher values improve recall but increase memory and build time.
    /// Typical: 16-64. Default: 32.
    pub m: usize,

    /// Maximum number of connections per node at higher layers.
    /// Usually m / 2. Default: 16.
    pub m_max: usize,

    /// Size of dynamic candidate list during construction.
    /// Higher values improve quality but slow construction.
    /// Typical: 100-500. Default: 200.
    pub ef_construction: usize,

    /// Size of dynamic candidate list during search.
    /// Higher values improve recall but slow queries.
    /// Can be tuned at query time. Default: 50.
    pub ef_search: usize,

    /// Level generation factor (1/ln(m)).
    /// Controls the probability of a node appearing at higher levels.
    pub ml: f64,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 32;
        Self {
            m,
            m_max: m / 2,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (m as f64).ln(),
            seed: None,
        }
    }
}

/// A node in the HNSW graph.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct HnswNode {
    /// Index into the original data array.
    data_idx: u32,
    /// Connections at each layer. neighbors[layer] = vec of neighbor indices.
    neighbors: Vec<Vec<u32>>,
    /// Maximum layer this node appears in (useful for debugging/stats).
    max_layer: usize,
}

/// Distance-index pair for priority queue operations.
#[derive(Debug, Clone, Copy)]
struct DistIdx {
    dist: f32,
    idx: u32,
}

impl PartialEq for DistIdx {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.idx == other.idx
    }
}

impl Eq for DistIdx {}

impl PartialOrd for DistIdx {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistIdx {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap by default (smaller distance = higher priority)
        other.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
    }
}

/// Reversed ordering for max-heap behavior.
#[derive(Debug, Clone, Copy)]
struct DistIdxMax(DistIdx);

impl PartialEq for DistIdxMax {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl Eq for DistIdxMax {}

impl PartialOrd for DistIdxMax {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistIdxMax {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap (larger distance = higher priority for removal)
        self.0.dist.partial_cmp(&other.0.dist).unwrap_or(Ordering::Equal)
    }
}

/// HNSW index for fast approximate nearest neighbor search on SDRs.
pub struct HnswIndex {
    config: HnswConfig,
    /// All nodes in the graph.
    nodes: Vec<HnswNode>,
    /// The fingerprints being indexed.
    fingerprints: Vec<Sdr>,
    /// Entry point (node index at highest layer).
    entry_point: Option<u32>,
    /// Current maximum layer in the graph.
    max_layer: usize,
    /// RNG for level generation.
    rng: ChaCha8Rng,
}

impl HnswIndex {
    /// Create a new empty HNSW index.
    pub fn new(config: HnswConfig) -> Self {
        let rng = match config.seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };

        Self {
            config,
            nodes: Vec::new(),
            fingerprints: Vec::new(),
            entry_point: None,
            max_layer: 0,
            rng,
        }
    }

    /// Build index from a collection of fingerprints.
    pub fn build(config: HnswConfig, fingerprints: Vec<Sdr>) -> Self {
        let mut index = Self::new(config);

        for fp in fingerprints {
            index.insert(fp);
        }

        index
    }

    /// Insert a single fingerprint into the index.
    pub fn insert(&mut self, fingerprint: Sdr) {
        let data_idx = self.fingerprints.len() as u32;
        self.fingerprints.push(fingerprint);

        // Generate random level for this node
        let level = self.random_level();

        // Create the node with empty neighbor lists
        let node = HnswNode {
            data_idx,
            neighbors: vec![Vec::new(); level + 1],
            max_layer: level,
        };

        let node_idx = self.nodes.len() as u32;
        self.nodes.push(node);

        // If this is the first node, make it the entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(node_idx);
            self.max_layer = level;
            return;
        }

        let entry_point = self.entry_point.unwrap();

        // Start from the entry point and traverse down
        let mut curr_obj = entry_point;
        let mut curr_dist = self.distance(data_idx, self.nodes[entry_point as usize].data_idx);

        // Traverse from top layer down to level+1, finding closest element
        for lc in (level + 1..=self.max_layer).rev() {
            let (new_obj, new_dist) = self.greedy_search_layer(data_idx, curr_obj, curr_dist, lc);
            curr_obj = new_obj;
            curr_dist = new_dist;
        }

        // For each layer from level down to 0
        for lc in (0..=level.min(self.max_layer)).rev() {
            // Search for ef_construction nearest neighbors
            let candidates = self.search_layer(data_idx, curr_obj, self.config.ef_construction, lc);

            // Select M neighbors using heuristic
            let neighbors = self.select_neighbors_heuristic(data_idx, &candidates, self.get_m(lc), lc);

            // Connect the new node to its neighbors
            self.nodes[node_idx as usize].neighbors[lc] = neighbors.clone();

            // Add bidirectional connections
            let max_conn = self.get_m(lc);
            for &neighbor_idx in &neighbors {
                // Check if this layer exists for neighbor and add connection
                if lc < self.nodes[neighbor_idx as usize].neighbors.len() {
                    self.nodes[neighbor_idx as usize].neighbors[lc].push(node_idx);

                    // Shrink if necessary
                    if self.nodes[neighbor_idx as usize].neighbors[lc].len() > max_conn {
                        let neighbor_data_idx = self.nodes[neighbor_idx as usize].data_idx;
                        let old_neighbors: Vec<u32> = self.nodes[neighbor_idx as usize].neighbors[lc].clone();
                        let candidates: Vec<DistIdx> = old_neighbors
                            .iter()
                            .map(|&n| DistIdx {
                                dist: self.distance(neighbor_data_idx, self.nodes[n as usize].data_idx),
                                idx: n,
                            })
                            .collect();
                        let new_neighbors = self.select_neighbors_heuristic(neighbor_data_idx, &candidates, max_conn, lc);
                        self.nodes[neighbor_idx as usize].neighbors[lc] = new_neighbors;
                    }
                }
            }

            // Update entry point for next layer
            if !candidates.is_empty() {
                curr_obj = candidates[0].idx;
            }
        }

        // Update entry point if new node has higher layer
        if level > self.max_layer {
            self.entry_point = Some(node_idx);
            self.max_layer = level;
        }
    }

    /// Search for k nearest neighbors to a query fingerprint.
    pub fn search(&self, query: &Sdr, k: usize) -> Vec<(u32, f32)> {
        self.search_ef(query, k, self.config.ef_search)
    }

    /// Search with custom ef parameter for recall/speed tradeoff.
    pub fn search_ef(&self, query: &Sdr, k: usize, ef: usize) -> Vec<(u32, f32)> {
        if self.nodes.is_empty() || self.entry_point.is_none() {
            return Vec::new();
        }

        let entry_point = self.entry_point.unwrap();
        let mut curr_obj = entry_point;
        let mut curr_dist = self.distance_to_query(query, self.nodes[entry_point as usize].data_idx);

        // Traverse from top layer down to layer 1
        for lc in (1..=self.max_layer).rev() {
            let (new_obj, new_dist) = self.greedy_search_layer_query(query, curr_obj, curr_dist, lc);
            curr_obj = new_obj;
            curr_dist = new_dist;
        }

        // Search layer 0 with ef candidates
        let candidates = self.search_layer_query(query, curr_obj, ef, 0);

        // Return top k
        candidates
            .into_iter()
            .take(k)
            .map(|di| (self.nodes[di.idx as usize].data_idx, di.dist))
            .collect()
    }

    /// Get the fingerprint at a given data index.
    pub fn get_fingerprint(&self, data_idx: u32) -> Option<&Sdr> {
        self.fingerprints.get(data_idx as usize)
    }

    /// Number of indexed fingerprints.
    pub fn len(&self) -> usize {
        self.fingerprints.len()
    }

    /// Check if index is empty.
    pub fn is_empty(&self) -> bool {
        self.fingerprints.is_empty()
    }

    /// Get index statistics.
    pub fn stats(&self) -> HnswStats {
        let mut layer_sizes = vec![0usize; self.max_layer + 1];
        let mut total_edges = 0usize;

        for node in &self.nodes {
            for (layer, neighbors) in node.neighbors.iter().enumerate() {
                layer_sizes[layer] += 1;
                total_edges += neighbors.len();
            }
        }

        HnswStats {
            num_nodes: self.nodes.len(),
            num_layers: self.max_layer + 1,
            layer_sizes,
            total_edges: total_edges / 2, // Each edge counted twice
            avg_degree: if self.nodes.is_empty() {
                0.0
            } else {
                total_edges as f64 / self.nodes.len() as f64
            },
        }
    }

    // --- Private methods ---

    /// Generate a random level for a new node.
    fn random_level(&mut self) -> usize {
        let r: f64 = self.rng.gen();
        let level = (-r.ln() * self.config.ml).floor() as usize;
        level.min(32) // Cap at reasonable maximum
    }

    /// Get maximum connections for a layer.
    fn get_m(&self, layer: usize) -> usize {
        if layer == 0 {
            self.config.m
        } else {
            self.config.m_max
        }
    }

    /// Compute distance between two indexed fingerprints.
    /// Uses 1 - Jaccard similarity as distance (0 = identical, 1 = completely different).
    #[inline]
    fn distance(&self, idx1: u32, idx2: u32) -> f32 {
        let fp1 = &self.fingerprints[idx1 as usize];
        let fp2 = &self.fingerprints[idx2 as usize];
        1.0 - fp1.jaccard_similarity(fp2) as f32
    }

    /// Compute distance between query and indexed fingerprint.
    #[inline]
    fn distance_to_query(&self, query: &Sdr, idx: u32) -> f32 {
        let fp = &self.fingerprints[idx as usize];
        1.0 - query.jaccard_similarity(fp) as f32
    }

    /// Greedy search within a single layer to find closest element.
    fn greedy_search_layer(&self, query_idx: u32, entry: u32, entry_dist: f32, layer: usize) -> (u32, f32) {
        let mut curr_obj = entry;
        let mut curr_dist = entry_dist;
        let mut changed = true;

        while changed {
            changed = false;

            let node = &self.nodes[curr_obj as usize];
            if layer >= node.neighbors.len() {
                break;
            }

            for &neighbor in &node.neighbors[layer] {
                let dist = self.distance(query_idx, self.nodes[neighbor as usize].data_idx);
                if dist < curr_dist {
                    curr_dist = dist;
                    curr_obj = neighbor;
                    changed = true;
                }
            }
        }

        (curr_obj, curr_dist)
    }

    /// Greedy search for external query.
    fn greedy_search_layer_query(&self, query: &Sdr, entry: u32, entry_dist: f32, layer: usize) -> (u32, f32) {
        let mut curr_obj = entry;
        let mut curr_dist = entry_dist;
        let mut changed = true;

        while changed {
            changed = false;

            let node = &self.nodes[curr_obj as usize];
            if layer >= node.neighbors.len() {
                break;
            }

            for &neighbor in &node.neighbors[layer] {
                let dist = self.distance_to_query(query, neighbor);
                if dist < curr_dist {
                    curr_dist = dist;
                    curr_obj = neighbor;
                    changed = true;
                }
            }
        }

        (curr_obj, curr_dist)
    }

    /// Search layer for ef nearest neighbors (during construction).
    fn search_layer(&self, query_idx: u32, entry: u32, ef: usize, layer: usize) -> Vec<DistIdx> {
        let entry_dist = self.distance(query_idx, self.nodes[entry as usize].data_idx);

        let mut visited = HashSet::new();
        visited.insert(entry);

        // Candidates (min-heap by distance)
        let mut candidates: BinaryHeap<DistIdx> = BinaryHeap::new();
        candidates.push(DistIdx { dist: entry_dist, idx: entry });

        // Result set (max-heap by distance for efficient pruning)
        let mut result: BinaryHeap<DistIdxMax> = BinaryHeap::new();
        result.push(DistIdxMax(DistIdx { dist: entry_dist, idx: entry }));

        while let Some(candidate) = candidates.pop() {
            // Get furthest result
            let furthest_dist = result.peek().map(|x| x.0.dist).unwrap_or(f32::MAX);

            // Stop if candidate is further than furthest result
            if candidate.dist > furthest_dist {
                break;
            }

            // Explore neighbors
            let node = &self.nodes[candidate.idx as usize];
            if layer >= node.neighbors.len() {
                continue;
            }

            for &neighbor in &node.neighbors[layer] {
                if visited.insert(neighbor) {
                    let dist = self.distance(query_idx, self.nodes[neighbor as usize].data_idx);
                    let furthest_dist = result.peek().map(|x| x.0.dist).unwrap_or(f32::MAX);

                    if dist < furthest_dist || result.len() < ef {
                        candidates.push(DistIdx { dist, idx: neighbor });
                        result.push(DistIdxMax(DistIdx { dist, idx: neighbor }));

                        if result.len() > ef {
                            result.pop();
                        }
                    }
                }
            }
        }

        // Convert to sorted vec
        let mut results: Vec<DistIdx> = result.into_iter().map(|x| x.0).collect();
        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        results
    }

    /// Search layer for external query.
    fn search_layer_query(&self, query: &Sdr, entry: u32, ef: usize, layer: usize) -> Vec<DistIdx> {
        let entry_dist = self.distance_to_query(query, entry);

        let mut visited = HashSet::new();
        visited.insert(entry);

        let mut candidates: BinaryHeap<DistIdx> = BinaryHeap::new();
        candidates.push(DistIdx { dist: entry_dist, idx: entry });

        let mut result: BinaryHeap<DistIdxMax> = BinaryHeap::new();
        result.push(DistIdxMax(DistIdx { dist: entry_dist, idx: entry }));

        while let Some(candidate) = candidates.pop() {
            let furthest_dist = result.peek().map(|x| x.0.dist).unwrap_or(f32::MAX);

            if candidate.dist > furthest_dist {
                break;
            }

            let node = &self.nodes[candidate.idx as usize];
            if layer >= node.neighbors.len() {
                continue;
            }

            for &neighbor in &node.neighbors[layer] {
                if visited.insert(neighbor) {
                    let dist = self.distance_to_query(query, neighbor);
                    let furthest_dist = result.peek().map(|x| x.0.dist).unwrap_or(f32::MAX);

                    if dist < furthest_dist || result.len() < ef {
                        candidates.push(DistIdx { dist, idx: neighbor });
                        result.push(DistIdxMax(DistIdx { dist, idx: neighbor }));

                        if result.len() > ef {
                            result.pop();
                        }
                    }
                }
            }
        }

        let mut results: Vec<DistIdx> = result.into_iter().map(|x| x.0).collect();
        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        results
    }

    /// Select neighbors using the heuristic from the HNSW paper.
    /// This prefers diverse neighbors over just the closest ones.
    fn select_neighbors_heuristic(&self, _query_idx: u32, candidates: &[DistIdx], m: usize, _layer: usize) -> Vec<u32> {
        if candidates.len() <= m {
            return candidates.iter().map(|c| c.idx).collect();
        }

        let mut result = Vec::with_capacity(m);
        let mut working: Vec<DistIdx> = candidates.to_vec();
        working.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));

        for candidate in working {
            if result.len() >= m {
                break;
            }

            // Check if this candidate is closer to query than to any selected neighbor
            let mut good = true;
            for &selected in &result {
                let dist_to_selected = self.distance(
                    self.nodes[candidate.idx as usize].data_idx,
                    self.nodes[selected as usize].data_idx,
                );
                if dist_to_selected < candidate.dist {
                    good = false;
                    break;
                }
            }

            if good {
                result.push(candidate.idx);
            }
        }

        // If heuristic didn't find enough, add closest remaining
        if result.len() < m {
            let sorted: Vec<DistIdx> = candidates.to_vec();
            for c in sorted {
                if !result.contains(&c.idx) {
                    result.push(c.idx);
                    if result.len() >= m {
                        break;
                    }
                }
            }
        }

        result
    }
}

/// Statistics about the HNSW index.
#[derive(Debug, Clone)]
pub struct HnswStats {
    /// Total number of nodes in the graph.
    pub num_nodes: usize,
    /// Number of layers.
    pub num_layers: usize,
    /// Number of nodes at each layer.
    pub layer_sizes: Vec<usize>,
    /// Total number of edges (undirected).
    pub total_edges: usize,
    /// Average node degree.
    pub avg_degree: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_fingerprints(n: usize, grid_size: u32) -> Vec<Sdr> {
        use rand::Rng;
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
    fn test_hnsw_build() {
        let fingerprints = create_test_fingerprints(100, 1024);
        let config = HnswConfig {
            m: 16,
            ef_construction: 100,
            ..Default::default()
        };

        let index = HnswIndex::build(config, fingerprints);
        assert_eq!(index.len(), 100);

        let stats = index.stats();
        assert!(stats.num_layers >= 1);
        assert!(stats.avg_degree > 0.0);
    }

    #[test]
    fn test_hnsw_search() {
        let fingerprints = create_test_fingerprints(500, 1024);
        let query = fingerprints[0].clone();

        let config = HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
            ..Default::default()
        };

        let index = HnswIndex::build(config, fingerprints);

        let results = index.search(&query, 10);
        assert_eq!(results.len(), 10);

        // First result should be the query itself (distance 0)
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 0.01);
    }

    #[test]
    fn test_hnsw_recall() {
        // Test that HNSW finds true nearest neighbors with high probability
        let fingerprints = create_test_fingerprints(1000, 1024);
        let query = fingerprints[500].clone();

        // Compute ground truth (brute force)
        let mut ground_truth: Vec<(u32, f32)> = fingerprints
            .iter()
            .enumerate()
            .map(|(i, fp)| (i as u32, 1.0 - query.jaccard_similarity(fp) as f32))
            .collect();
        ground_truth.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let ground_truth_10: HashSet<u32> = ground_truth.iter().take(10).map(|x| x.0).collect();

        // HNSW search
        let config = HnswConfig {
            m: 32,
            ef_construction: 200,
            ef_search: 100,
            seed: Some(42),
            ..Default::default()
        };
        let index = HnswIndex::build(config, fingerprints);
        let results = index.search(&query, 10);

        // Check recall
        let hnsw_results: HashSet<u32> = results.iter().map(|x| x.0).collect();
        let recall = ground_truth_10.intersection(&hnsw_results).count() as f64 / 10.0;

        // HNSW should achieve at least 80% recall with these parameters
        assert!(recall >= 0.8, "Recall {} is too low", recall);
    }
}
