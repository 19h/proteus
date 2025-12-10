//! SOM training algorithms.
//!
//! This module provides efficient SOM training using compact word embeddings
//! learned through a skip-gram-like approach during a single pass over the corpus.

use crate::config::SomConfig;
use crate::error::{ProteusError, Result};
use crate::som::Som;
use log::info;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::collections::HashMap;

/// Default embedding dimension for word vectors.
pub const DEFAULT_EMBEDDING_DIM: usize = 100;

/// Context for training samples.
#[derive(Debug, Clone)]
pub struct TrainingContext {
    /// The center word.
    pub center: String,
    /// Context embedding vector.
    pub embedding: Vec<f64>,
}

impl TrainingContext {
    /// Creates a new training context.
    pub fn new(center: String, embedding: Vec<f64>) -> Self {
        Self { center, embedding }
    }
}

/// Compact word embeddings learned from co-occurrence.
///
/// Instead of vocabulary-sized BoW vectors, this uses a fixed-dimension
/// embedding space where words are represented by their co-occurrence patterns
/// projected into a low-dimensional space via random projection (a simple
/// but effective dimensionality reduction technique).
pub struct WordEmbeddings {
    embeddings: HashMap<String, Vec<f64>>,
    dim: usize,
}

impl WordEmbeddings {
    /// Learn word embeddings from context windows.
    ///
    /// Uses random projection: each unique context word contributes a fixed
    /// random vector, and a word's embedding is the sum of its context vectors.
    pub fn from_contexts(
        contexts: &[(String, Vec<String>)],
        dim: usize,
        seed: Option<u64>,
    ) -> Self {
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };

        // Collect all unique words
        let mut all_words: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for (center, ctx) in contexts {
            all_words.insert(center);
            for w in ctx {
                all_words.insert(w);
            }
        }

        // Generate random projection vectors for each word
        use rand::Rng;
        let random_vecs: HashMap<String, Vec<f64>> = all_words
            .iter()
            .map(|&w| {
                let vec: Vec<f64> = (0..dim)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect();
                (w.to_string(), vec)
            })
            .collect();

        // Build embeddings by summing context vectors
        let mut word_contexts: HashMap<String, Vec<f64>> = HashMap::new();
        let mut word_counts: HashMap<String, usize> = HashMap::new();

        for (center, ctx) in contexts {
            let entry = word_contexts
                .entry(center.clone())
                .or_insert_with(|| vec![0.0; dim]);

            for ctx_word in ctx {
                if let Some(ctx_vec) = random_vecs.get(ctx_word) {
                    for (i, v) in ctx_vec.iter().enumerate() {
                        entry[i] += v;
                    }
                }
            }
            *word_counts.entry(center.clone()).or_insert(0) += 1;
        }

        // Normalize embeddings
        let embeddings: HashMap<String, Vec<f64>> = word_contexts
            .into_iter()
            .map(|(word, mut vec)| {
                let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 0.0 {
                    for v in &mut vec {
                        *v /= norm;
                    }
                }
                (word, vec)
            })
            .collect();

        Self { embeddings, dim }
    }

    /// Get embedding for a word.
    pub fn get(&self, word: &str) -> Option<&Vec<f64>> {
        self.embeddings.get(word)
    }

    /// Get embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of words with embeddings.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }
}

/// SOM trainer with configurable hyperparameters.
pub struct SomTrainer {
    config: SomConfig,
    rng: ChaCha8Rng,
}

impl SomTrainer {
    /// Creates a new trainer with the given configuration.
    pub fn new(config: SomConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::from_entropy(),
        };

        Self { config, rng }
    }

    /// Computes the learning rate at a given iteration.
    #[inline]
    pub fn learning_rate(&self, iteration: usize) -> f64 {
        let t = iteration as f64 / self.config.iterations as f64;
        let initial = self.config.initial_learning_rate;
        let final_lr = self.config.final_learning_rate;
        initial * (final_lr / initial).powf(t)
    }

    /// Computes the neighborhood radius at a given iteration.
    #[inline]
    pub fn radius(&self, iteration: usize) -> f64 {
        let t = iteration as f64 / self.config.iterations as f64;
        let initial = self.config.initial_radius;
        let final_r = self.config.final_radius;
        initial * (final_r / initial).powf(t)
    }

    /// Fast training using batch processing with parallelism.
    ///
    /// Uses a two-phase approach:
    /// 1. Quick coarse pass with large radius to establish topology
    /// 2. Fine-tuning pass with small radius for precise mapping
    pub fn train_fast(
        &mut self,
        som: &mut Som,
        embeddings: &WordEmbeddings,
        contexts: &[(String, Vec<String>)],
    ) -> Result<HashMap<String, Vec<usize>>> {
        if contexts.is_empty() {
            return Err(ProteusError::Training("No training contexts provided".to_string()));
        }

        // Pre-compute all context embeddings
        let context_vecs: Vec<(String, Vec<f64>)> = contexts
            .par_iter()
            .filter_map(|(center, ctx)| {
                let mut vec = vec![0.0; embeddings.dim()];
                let mut count = 0;
                for w in ctx {
                    if let Some(e) = embeddings.get(w) {
                        for (i, v) in e.iter().enumerate() {
                            vec[i] += v;
                        }
                        count += 1;
                    }
                }
                if count > 0 {
                    let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
                    if norm > 0.0 {
                        for v in &mut vec {
                            *v /= norm;
                        }
                    }
                    Some((center.clone(), vec))
                } else {
                    None
                }
            })
            .collect();

        if context_vecs.is_empty() {
            return Err(ProteusError::Training("No valid context vectors".to_string()));
        }

        let dim = som.dimension;
        let weight_dim = embeddings.dim();
        let num_neurons = som.neurons.len();

        // Flatten neuron weights for cache-friendly access
        let mut neuron_weights: Vec<f64> = Vec::with_capacity(num_neurons * weight_dim);
        for neuron in &som.neurons {
            neuron_weights.extend_from_slice(&neuron.weights);
        }

        info!(
            "Training SOM: {} samples, {} neurons, {} dim",
            context_vecs.len(), num_neurons, weight_dim
        );

        // Sample subset for coarse training (10% or 10k, whichever is smaller)
        let coarse_sample_size = (context_vecs.len() / 10).min(10000).max(1000);
        let mut indices: Vec<usize> = (0..context_vecs.len()).collect();
        indices.shuffle(&mut self.rng);
        let coarse_indices = &indices[..coarse_sample_size];

        // Phase 1: Coarse topology (large radius, high learning rate)
        info!("Phase 1: Coarse topology ({} samples, radius=16)", coarse_sample_size);
        let coarse_lr = 0.1;
        let coarse_radius = 16.0;
        let radius_int = (coarse_radius * 1.5f64).ceil() as i32;

        for &idx in coarse_indices {
            let (_, input) = &context_vecs[idx];
            let bmu_idx = find_bmu_flat_seq(&neuron_weights, input, num_neurons, weight_dim);

            let bmu_row = bmu_idx / dim;
            let bmu_col = bmu_idx % dim;

            for dr in -radius_int..=radius_int {
                for dc in -radius_int..=radius_int {
                    let grid_dist_sq = (dr * dr + dc * dc) as f64;
                    if grid_dist_sq > coarse_radius * coarse_radius * 4.0 { continue; }

                    let bmu_r = bmu_row as i32;
                    let bmu_c = bmu_col as i32;
                    let dim_i = dim as i32;
                    let nr = if som.toroidal {
                        ((bmu_r + dr).rem_euclid(dim_i)) as usize
                    } else {
                        let r = bmu_r + dr;
                        if r < 0 || r >= dim_i { continue; }
                        r as usize
                    };
                    let nc = if som.toroidal {
                        ((bmu_c + dc).rem_euclid(dim_i)) as usize
                    } else {
                        let c = bmu_c + dc;
                        if c < 0 || c >= dim_i { continue; }
                        c as usize
                    };

                    let neuron_idx = nr * dim + nc;
                    let sigma_sq = coarse_radius * coarse_radius;
                    let neighborhood = (-grid_dist_sq / (2.0 * sigma_sq)).exp();

                    if neighborhood > 0.01 {
                        let influence = coarse_lr * neighborhood;
                        let offset = neuron_idx * weight_dim;
                        for i in 0..weight_dim {
                            neuron_weights[offset + i] += influence * (input[i] - neuron_weights[offset + i]);
                        }
                    }
                }
            }
        }

        // Phase 2: Fine mapping (all samples, small radius) - parallel BMU, collect results
        info!("Phase 2: Fine mapping ({} samples, radius=2)", context_vecs.len());
        let fine_lr = 0.02;
        let fine_radius = 2.0;
        let fine_radius_int = (fine_radius * 1.5f64).ceil() as i32;

        // Find all BMUs in parallel
        let bmus: Vec<(usize, usize)> = context_vecs
            .par_iter()
            .enumerate()
            .map(|(sample_idx, (_, input))| {
                let bmu = find_bmu_flat_seq(&neuron_weights, input, num_neurons, weight_dim);
                (sample_idx, bmu)
            })
            .collect();

        // Record BMUs
        let mut word_to_bmus: HashMap<String, Vec<usize>> = HashMap::new();
        for (sample_idx, bmu_idx) in &bmus {
            let (center, _) = &context_vecs[*sample_idx];
            word_to_bmus
                .entry(center.clone())
                .or_default()
                .push(*bmu_idx);
        }

        // Apply fine updates (with small radius this is fast)
        for (sample_idx, bmu_idx) in &bmus {
            let (_, input) = &context_vecs[*sample_idx];
            let bmu_row = bmu_idx / dim;
            let bmu_col = bmu_idx % dim;

            for dr in -fine_radius_int..=fine_radius_int {
                for dc in -fine_radius_int..=fine_radius_int {
                    let grid_dist_sq = (dr * dr + dc * dc) as f64;
                    if grid_dist_sq > fine_radius * fine_radius * 4.0 { continue; }

                    let bmu_r = bmu_row as i32;
                    let bmu_c = bmu_col as i32;
                    let dim_i = dim as i32;
                    let nr = if som.toroidal {
                        ((bmu_r + dr).rem_euclid(dim_i)) as usize
                    } else {
                        let r = bmu_r + dr;
                        if r < 0 || r >= dim_i { continue; }
                        r as usize
                    };
                    let nc = if som.toroidal {
                        ((bmu_c + dc).rem_euclid(dim_i)) as usize
                    } else {
                        let c = bmu_c + dc;
                        if c < 0 || c >= dim_i { continue; }
                        c as usize
                    };

                    let neuron_idx = nr * dim + nc;
                    let sigma_sq = fine_radius * fine_radius;
                    let neighborhood = (-grid_dist_sq / (2.0 * sigma_sq)).exp();

                    if neighborhood > 0.01 {
                        let influence = fine_lr * neighborhood;
                        let offset = neuron_idx * weight_dim;
                        for i in 0..weight_dim {
                            neuron_weights[offset + i] += influence * (input[i] - neuron_weights[offset + i]);
                        }
                    }
                }
            }
        }

        // Copy weights back to neurons
        for (i, neuron) in som.neurons.iter_mut().enumerate() {
            let offset = i * weight_dim;
            neuron.weights.copy_from_slice(&neuron_weights[offset..offset + weight_dim]);
        }

        info!("Training completed. {} unique words mapped.", word_to_bmus.len());
        Ok(word_to_bmus)
    }

    /// Original train method for backwards compatibility.
    pub fn train(
        &mut self,
        som: &mut Som,
        contexts: &[TrainingContext],
    ) -> Result<HashMap<String, Vec<usize>>> {
        if contexts.is_empty() {
            return Err(ProteusError::Training("No training contexts provided".to_string()));
        }

        let mut word_to_bmus: HashMap<String, Vec<usize>> = HashMap::new();
        let mut shuffled_indices: Vec<usize> = (0..contexts.len()).collect();

        info!(
            "Starting SOM training with {} iterations on {} contexts",
            self.config.iterations,
            contexts.len()
        );

        let dim = som.dimension;

        for iteration in 0..self.config.iterations {
            if iteration % contexts.len() == 0 {
                shuffled_indices.shuffle(&mut self.rng);
            }

            let idx = shuffled_indices[iteration % contexts.len()];
            let context = &contexts[idx];

            // Find BMU
            let bmu_idx = som.neurons
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let dist_a = a.distance_squared(&context.embedding);
                    let dist_b = b.distance_squared(&context.embedding);
                    dist_a.partial_cmp(&dist_b).unwrap()
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            word_to_bmus
                .entry(context.center.clone())
                .or_default()
                .push(bmu_idx);

            let lr = self.learning_rate(iteration);
            let radius = self.radius(iteration);

            // Optimized update - only neurons within radius
            let bmu_row = bmu_idx / dim;
            let bmu_col = bmu_idx % dim;
            let radius_int = (radius * 2.0).ceil() as i32;

            for dr in -radius_int..=radius_int {
                for dc in -radius_int..=radius_int {
                    let nr = if som.toroidal {
                        ((bmu_row as i32 + dr).rem_euclid(dim as i32)) as usize
                    } else {
                        let r = bmu_row as i32 + dr;
                        if r < 0 || r >= dim as i32 { continue; }
                        r as usize
                    };
                    let nc = if som.toroidal {
                        ((bmu_col as i32 + dc).rem_euclid(dim as i32)) as usize
                    } else {
                        let c = bmu_col as i32 + dc;
                        if c < 0 || c >= dim as i32 { continue; }
                        c as usize
                    };

                    let neuron_idx = nr * dim + nc;
                    let grid_dist_sq = (dr * dr + dc * dc) as f64;
                    let sigma_sq = radius * radius;
                    let neighborhood = (-grid_dist_sq / (2.0 * sigma_sq)).exp();

                    if neighborhood > 0.001 {
                        let neuron = &mut som.neurons[neuron_idx];
                        let influence = lr * neighborhood;
                        for (i, w) in neuron.weights.iter_mut().enumerate() {
                            *w += influence * (context.embedding[i] - *w);
                        }
                    }
                }
            }

            if iteration % 10000 == 0 || iteration == self.config.iterations - 1 {
                info!(
                    "Iteration {}/{}: lr={:.4}, radius={:.2}",
                    iteration, self.config.iterations, lr, radius
                );
            }
        }

        info!("SOM training completed");
        Ok(word_to_bmus)
    }

    /// Converts word-to-BMUs mapping into stable fingerprints.
    pub fn compute_fingerprints(
        word_to_bmus: &HashMap<String, Vec<usize>>,
        target_bits: usize,
    ) -> HashMap<String, Vec<usize>> {
        let mut fingerprints = HashMap::new();

        for (word, bmus) in word_to_bmus {
            let mut bmu_counts: HashMap<usize, usize> = HashMap::new();
            for &bmu in bmus {
                *bmu_counts.entry(bmu).or_insert(0) += 1;
            }

            let mut sorted_bmus: Vec<(usize, usize)> = bmu_counts.into_iter().collect();
            sorted_bmus.sort_by(|a, b| b.1.cmp(&a.1));

            let fingerprint: Vec<usize> = sorted_bmus
                .into_iter()
                .take(target_bits)
                .map(|(bmu, _)| bmu)
                .collect();

            fingerprints.insert(word.clone(), fingerprint);
        }

        fingerprints
    }
}

/// Find BMU using flat weight array (cache-friendly, sequential).
/// This is used inside parallel iterators where each thread finds BMU for its sample.
#[inline]
fn find_bmu_flat_seq(weights: &[f64], input: &[f64], num_neurons: usize, weight_dim: usize) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f64::MAX;

    for i in 0..num_neurons {
        let offset = i * weight_dim;
        let mut dist = 0.0;
        for j in 0..weight_dim {
            let diff = input[j] - weights[offset + j];
            dist += diff * diff;
        }
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SomConfig {
        SomConfig {
            dimension: 8,
            weight_dimension: 10,
            iterations: 100,
            initial_learning_rate: 0.1,
            final_learning_rate: 0.01,
            initial_radius: 4.0,
            final_radius: 1.0,
            seed: Some(42),
            ..Default::default()
        }
    }

    #[test]
    fn test_learning_rate_decay() {
        let trainer = SomTrainer::new(test_config());

        let initial = trainer.learning_rate(0);
        let final_lr = trainer.learning_rate(99);

        assert!((initial - 0.1).abs() < 1e-6);
        assert!(final_lr < initial);
        assert!(final_lr > 0.01);
    }

    #[test]
    fn test_radius_decay() {
        let trainer = SomTrainer::new(test_config());

        let initial = trainer.radius(0);
        let final_r = trainer.radius(99);

        assert!((initial - 4.0).abs() < 1e-6);
        assert!(final_r < initial);
        assert!(final_r > 1.0);
    }

    #[test]
    fn test_train_simple() {
        let config = test_config();
        let mut som = Som::new(&config);
        let mut trainer = SomTrainer::new(config);

        let contexts = vec![
            TrainingContext::new("hello".to_string(), vec![1.0; 10]),
            TrainingContext::new("world".to_string(), vec![0.0; 10]),
        ];

        let result = trainer.train(&mut som, &contexts);
        assert!(result.is_ok());

        let word_to_bmus = result.unwrap();
        assert!(word_to_bmus.contains_key("hello"));
        assert!(word_to_bmus.contains_key("world"));
    }

    #[test]
    fn test_compute_fingerprints() {
        let mut word_to_bmus = HashMap::new();
        word_to_bmus.insert("test".to_string(), vec![1, 1, 2, 3, 3, 3, 4]);

        let fingerprints = SomTrainer::compute_fingerprints(&word_to_bmus, 3);

        assert!(fingerprints.contains_key("test"));
        let fp = &fingerprints["test"];
        assert_eq!(fp.len(), 3);
        assert_eq!(fp[0], 3); // Most frequent
    }

    #[test]
    fn test_word_embeddings() {
        let contexts = vec![
            ("hello".to_string(), vec!["world".to_string(), "there".to_string()]),
            ("world".to_string(), vec!["hello".to_string(), "peace".to_string()]),
        ];

        let embeddings = WordEmbeddings::from_contexts(&contexts, 50, Some(42));

        assert!(embeddings.get("hello").is_some());
        assert!(embeddings.get("world").is_some());
        assert_eq!(embeddings.dim(), 50);
    }
}
