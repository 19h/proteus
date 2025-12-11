//! SOM training algorithms.
//!
//! This module provides efficient SOM training using compact word embeddings
//! learned through a skip-gram-like approach during a single pass over the corpus.
//!
//! Performance optimizations:
//! - f32 instead of f64 (2x SIMD throughput, half memory bandwidth)
//! - Flat weight arrays for cache-friendly access
//! - Pre-computed neighborhood weights
//! - Parallel BMU search using rayon
//! - SIMD-friendly loop unrolling

use crate::config::SomConfig;
use crate::error::{ProteusError, Result};
use crate::som::simd::{
    add_vectors_f32, find_all_bmus_parallel, find_all_bmus_parallel_fast, normalize_f32,
    precompute_neighborhood, update_weights_f32,
};
use crate::som::Som;
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::collections::HashMap;

/// Default embedding dimension for word vectors.
pub const DEFAULT_EMBEDDING_DIM: usize = 100;

/// Format large numbers with commas for readability.
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Context for training samples (f64 version for backwards compatibility).
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
/// Uses f32 for SIMD efficiency. Instead of vocabulary-sized BoW vectors,
/// this uses a fixed-dimension embedding space where words are represented
/// by their co-occurrence patterns projected into a low-dimensional space
/// via random projection.
pub struct WordEmbeddings {
    embeddings: HashMap<String, Vec<f32>>,
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

        // Generate random projection vectors for each word (f32)
        use rand::Rng;
        let random_vecs: HashMap<String, Vec<f32>> = all_words
            .iter()
            .map(|&w| {
                let vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
                (w.to_string(), vec)
            })
            .collect();

        // Build embeddings by summing context vectors using SIMD
        let mut word_contexts: HashMap<String, Vec<f32>> = HashMap::new();

        for (center, ctx) in contexts {
            let entry = word_contexts
                .entry(center.clone())
                .or_insert_with(|| vec![0.0f32; dim]);

            for ctx_word in ctx {
                if let Some(ctx_vec) = random_vecs.get(ctx_word) {
                    add_vectors_f32(entry, ctx_vec);
                }
            }
        }

        // Normalize embeddings using SIMD-friendly function
        let embeddings: HashMap<String, Vec<f32>> = word_contexts
            .into_iter()
            .map(|(word, mut vec)| {
                normalize_f32(&mut vec);
                (word, vec)
            })
            .collect();

        Self { embeddings, dim }
    }

    /// Get embedding for a word.
    pub fn get(&self, word: &str) -> Option<&Vec<f32>> {
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

    /// Ultra-fast training using f32 SIMD operations.
    ///
    /// Uses a two-phase approach:
    /// 1. Quick coarse pass with medium radius to establish topology
    /// 2. Fine-tuning pass with small radius for precise mapping
    pub fn train_fast(
        &mut self,
        som: &mut Som,
        embeddings: &WordEmbeddings,
        contexts: &[(String, Vec<String>)],
    ) -> Result<HashMap<String, Vec<usize>>> {
        if contexts.is_empty() {
            return Err(ProteusError::Training(
                "No training contexts provided".to_string(),
            ));
        }

        // Pre-compute all context embeddings as f32 using SIMD vector addition
        let context_vecs: Vec<(String, Vec<f32>)> = contexts
            .par_iter()
            .filter_map(|(center, ctx)| {
                let mut vec = vec![0.0f32; embeddings.dim()];
                let mut count = 0;
                for w in ctx {
                    if let Some(e) = embeddings.get(w) {
                        add_vectors_f32(&mut vec, e);
                        count += 1;
                    }
                }
                if count > 0 {
                    normalize_f32(&mut vec);
                    Some((center.clone(), vec))
                } else {
                    None
                }
            })
            .collect();

        if context_vecs.is_empty() {
            return Err(ProteusError::Training(
                "No valid context vectors".to_string(),
            ));
        }

        let dim = som.dimension;
        let weight_dim = embeddings.dim();
        let num_neurons = som.neurons.len();

        // Convert neuron weights to f32 flat array for SIMD operations
        let mut neuron_weights: Vec<f32> = Vec::with_capacity(num_neurons * weight_dim);
        for neuron in &som.neurons {
            for &w in &neuron.weights {
                neuron_weights.push(w as f32);
            }
        }

        info!(
            "Training SOM: {} samples, {} neurons, {} dim (f32 SIMD)",
            context_vecs.len(),
            num_neurons,
            weight_dim
        );

        // Sample subset for coarse training (10% or 20k, whichever is smaller)
        let coarse_sample_size = (context_vecs.len() / 10).min(20000).max(2000);
        let mut indices: Vec<usize> = (0..context_vecs.len()).collect();
        indices.shuffle(&mut self.rng);
        let coarse_indices: Vec<usize> = indices[..coarse_sample_size].to_vec();

        // Phase 1: Coarse topology (large radius for good organization)
        let coarse_lr = 0.1f32;
        let coarse_radius = 12.0f32;

        // Pre-compute neighborhood weights for coarse phase (lower threshold = more neighbors)
        let coarse_neighbors = precompute_neighborhood(coarse_radius, 0.02);
        info!(
            "Phase 1: Coarse topology ({} samples, {} neighbors)",
            coarse_sample_size,
            coarse_neighbors.len()
        );

        // Find all coarse BMUs in parallel first
        let coarse_inputs: Vec<(usize, Vec<f32>)> = coarse_indices
            .iter()
            .map(|&i| (i, context_vecs[i].1.clone()))
            .collect();
        let coarse_bmus =
            find_all_bmus_parallel(&neuron_weights, &coarse_inputs, num_neurons, weight_dim);

        // Apply coarse updates sequentially (needed for weight consistency)
        for (sample_idx, bmu_idx) in coarse_bmus {
            let input = &context_vecs[sample_idx].1;
            let bmu_row = (bmu_idx / dim) as i32;
            let bmu_col = (bmu_idx % dim) as i32;
            let dim_i = dim as i32;

            for &(dr, dc, weight) in &coarse_neighbors {
                let nr = if som.toroidal {
                    (bmu_row + dr).rem_euclid(dim_i) as usize
                } else {
                    let r = bmu_row + dr;
                    if r < 0 || r >= dim_i {
                        continue;
                    }
                    r as usize
                };
                let nc = if som.toroidal {
                    (bmu_col + dc).rem_euclid(dim_i) as usize
                } else {
                    let c = bmu_col + dc;
                    if c < 0 || c >= dim_i {
                        continue;
                    }
                    c as usize
                };

                let neuron_idx = nr * dim + nc;
                let influence = coarse_lr * weight;
                let offset = neuron_idx * weight_dim;
                update_weights_f32(
                    &mut neuron_weights[offset..offset + weight_dim],
                    input,
                    influence,
                );
            }
        }

        // Phase 2: Fine-tuning with smaller subset (small radius updates)
        let fine_sample_size = (context_vecs.len() / 5).min(50000).max(5000);
        let fine_lr = 0.05f32;
        let fine_radius = 4.0f32;
        let fine_neighbors = precompute_neighborhood(fine_radius, 0.05);

        // Sample for fine tuning
        indices.shuffle(&mut self.rng);
        let fine_indices: Vec<usize> = indices[..fine_sample_size].to_vec();

        info!(
            "Phase 2: Fine-tuning ({} samples, {} neighbors)",
            fine_sample_size,
            fine_neighbors.len()
        );

        // Find BMUs in parallel for fine samples
        let fine_inputs: Vec<(usize, Vec<f32>)> = fine_indices
            .iter()
            .map(|&i| (i, context_vecs[i].1.clone()))
            .collect();
        let fine_bmus =
            find_all_bmus_parallel(&neuron_weights, &fine_inputs, num_neurons, weight_dim);

        // Apply fine updates
        for (sample_idx, bmu_idx) in fine_bmus {
            let input = &context_vecs[sample_idx].1;
            let bmu_row = (bmu_idx / dim) as i32;
            let bmu_col = (bmu_idx % dim) as i32;
            let dim_i = dim as i32;

            for &(dr, dc, weight) in &fine_neighbors {
                let nr = if som.toroidal {
                    (bmu_row + dr).rem_euclid(dim_i) as usize
                } else {
                    let r = bmu_row + dr;
                    if r < 0 || r >= dim_i {
                        continue;
                    }
                    r as usize
                };
                let nc = if som.toroidal {
                    (bmu_col + dc).rem_euclid(dim_i) as usize
                } else {
                    let c = bmu_col + dc;
                    if c < 0 || c >= dim_i {
                        continue;
                    }
                    c as usize
                };

                let neuron_idx = nr * dim + nc;
                let influence = fine_lr * weight;
                let offset = neuron_idx * weight_dim;
                update_weights_f32(
                    &mut neuron_weights[offset..offset + weight_dim],
                    input,
                    influence,
                );
            }
        }

        // Phase 3: Final BMU assignment (all samples)
        info!(
            "Phase 3: Final BMU assignment ({} samples)",
            context_vecs.len()
        );

        // Prepare indexed inputs for parallel BMU search
        let indexed_inputs: Vec<(usize, Vec<f32>)> = context_vecs
            .iter()
            .enumerate()
            .map(|(i, (_, v))| (i, v.clone()))
            .collect();

        // Find all BMUs in parallel using SIMD
        let bmus = find_all_bmus_parallel(&neuron_weights, &indexed_inputs, num_neurons, weight_dim);

        // Record BMUs (this is what we return)
        let mut word_to_bmus: HashMap<String, Vec<usize>> = HashMap::new();
        for &(sample_idx, bmu_idx) in &bmus {
            let center = &context_vecs[sample_idx].0;
            word_to_bmus
                .entry(center.clone())
                .or_default()
                .push(bmu_idx);
        }

        // Copy f32 weights back to neurons as f64
        for (i, neuron) in som.neurons.iter_mut().enumerate() {
            let offset = i * weight_dim;
            for (j, w) in neuron.weights.iter_mut().enumerate() {
                *w = neuron_weights[offset + j] as f64;
            }
        }

        info!(
            "Training completed. {} unique words mapped.",
            word_to_bmus.len()
        );
        Ok(word_to_bmus)
    }

    /// Ultra-fast training with progress display.
    ///
    /// Same as `train_fast` but shows progress bars for each phase.
    pub fn train_fast_with_progress<F>(
        &mut self,
        som: &mut Som,
        embeddings: &WordEmbeddings,
        contexts: &[(String, Vec<String>)],
        _callback: F,
    ) -> Result<HashMap<String, Vec<usize>>>
    where
        F: Fn(usize, usize, usize, &str),
    {
        if contexts.is_empty() {
            return Err(ProteusError::Training(
                "No training contexts provided".to_string(),
            ));
        }

        let bar_style = ProgressStyle::default_bar()
            .template("  {msg}\n  [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) ETA: {eta}")
            .unwrap()
            .progress_chars("█▓▒░  ");

        let _spinner_style = ProgressStyle::default_spinner()
            .template("  {spinner:.cyan} {msg}")
            .unwrap();

        // Helper to print completion messages that persist
        let print_done = |msg: &str| {
            println!("  ✓ {}", msg);
        };

        // Pre-compute all context embeddings with progress
        let pb = ProgressBar::new(contexts.len() as u64);
        pb.set_style(bar_style.clone());
        pb.set_message("Computing context embeddings...");

        let context_vecs: Vec<(String, Vec<f32>)> = contexts
            .par_iter()
            .filter_map(|(center, ctx)| {
                let mut vec = vec![0.0f32; embeddings.dim()];
                let mut count = 0;
                for w in ctx {
                    if let Some(e) = embeddings.get(w) {
                        add_vectors_f32(&mut vec, e);
                        count += 1;
                    }
                }
                if count > 0 {
                    normalize_f32(&mut vec);
                    Some((center.clone(), vec))
                } else {
                    None
                }
            })
            .inspect(|_| pb.inc(1))
            .collect();

        pb.finish_and_clear();
        print_done(&format!("Computed {} context embeddings", format_number(context_vecs.len())));

        if context_vecs.is_empty() {
            return Err(ProteusError::Training(
                "No valid context vectors".to_string(),
            ));
        }

        let dim = som.dimension;
        let weight_dim = embeddings.dim();
        let num_neurons = som.neurons.len();

        // Convert neuron weights to f32 flat array for SIMD operations
        let mut neuron_weights: Vec<f32> = Vec::with_capacity(num_neurons * weight_dim);
        for neuron in &som.neurons {
            for &w in &neuron.weights {
                neuron_weights.push(w as f32);
            }
        }

        // Phase 1: Coarse topology
        let coarse_sample_size = (context_vecs.len() / 10).min(20000).max(2000);
        let coarse_lr = 0.1f32;
        let coarse_radius = 12.0f32;
        let coarse_neighbors = precompute_neighborhood(coarse_radius, 0.02);

        let mut indices: Vec<usize> = (0..context_vecs.len()).collect();
        indices.shuffle(&mut self.rng);
        let coarse_indices: Vec<usize> = indices[..coarse_sample_size].to_vec();

        let pb = ProgressBar::new(coarse_sample_size as u64);
        pb.set_style(bar_style.clone());
        pb.set_message(format!("Phase 1: Coarse topology ({} neighbors)", coarse_neighbors.len()));

        // Find all coarse BMUs in parallel
        let coarse_inputs: Vec<(usize, Vec<f32>)> = coarse_indices
            .iter()
            .map(|&i| (i, context_vecs[i].1.clone()))
            .collect();
        let coarse_bmus =
            find_all_bmus_parallel(&neuron_weights, &coarse_inputs, num_neurons, weight_dim);

        // Apply coarse updates sequentially
        for (i, (sample_idx, bmu_idx)) in coarse_bmus.iter().enumerate() {
            let input = &context_vecs[*sample_idx].1;
            let bmu_row = (*bmu_idx / dim) as i32;
            let bmu_col = (*bmu_idx % dim) as i32;
            let dim_i = dim as i32;

            for &(dr, dc, weight) in &coarse_neighbors {
                let nr = if som.toroidal {
                    (bmu_row + dr).rem_euclid(dim_i) as usize
                } else {
                    let r = bmu_row + dr;
                    if r < 0 || r >= dim_i {
                        continue;
                    }
                    r as usize
                };
                let nc = if som.toroidal {
                    (bmu_col + dc).rem_euclid(dim_i) as usize
                } else {
                    let c = bmu_col + dc;
                    if c < 0 || c >= dim_i {
                        continue;
                    }
                    c as usize
                };

                let neuron_idx = nr * dim + nc;
                let influence = coarse_lr * weight;
                let offset = neuron_idx * weight_dim;
                update_weights_f32(
                    &mut neuron_weights[offset..offset + weight_dim],
                    input,
                    influence,
                );
            }

            if i % 500 == 0 {
                pb.set_position(i as u64);
            }
        }
        pb.finish_and_clear();
        print_done(&format!("Phase 1: Coarse topology ({} samples)", coarse_sample_size));

        // Phase 2: Fine-tuning
        let fine_sample_size = (context_vecs.len() / 5).min(50000).max(5000);
        let fine_lr = 0.05f32;
        let fine_radius = 4.0f32;
        let fine_neighbors = precompute_neighborhood(fine_radius, 0.05);

        indices.shuffle(&mut self.rng);
        let fine_indices: Vec<usize> = indices[..fine_sample_size].to_vec();

        let pb = ProgressBar::new(fine_sample_size as u64);
        pb.set_style(bar_style.clone());
        pb.set_message(format!("Phase 2: Fine-tuning ({} neighbors)", fine_neighbors.len()));

        let fine_inputs: Vec<(usize, Vec<f32>)> = fine_indices
            .iter()
            .map(|&i| (i, context_vecs[i].1.clone()))
            .collect();
        let fine_bmus =
            find_all_bmus_parallel(&neuron_weights, &fine_inputs, num_neurons, weight_dim);

        for (i, (sample_idx, bmu_idx)) in fine_bmus.iter().enumerate() {
            let input = &context_vecs[*sample_idx].1;
            let bmu_row = (*bmu_idx / dim) as i32;
            let bmu_col = (*bmu_idx % dim) as i32;
            let dim_i = dim as i32;

            for &(dr, dc, weight) in &fine_neighbors {
                let nr = if som.toroidal {
                    (bmu_row + dr).rem_euclid(dim_i) as usize
                } else {
                    let r = bmu_row + dr;
                    if r < 0 || r >= dim_i {
                        continue;
                    }
                    r as usize
                };
                let nc = if som.toroidal {
                    (bmu_col + dc).rem_euclid(dim_i) as usize
                } else {
                    let c = bmu_col + dc;
                    if c < 0 || c >= dim_i {
                        continue;
                    }
                    c as usize
                };

                let neuron_idx = nr * dim + nc;
                let influence = fine_lr * weight;
                let offset = neuron_idx * weight_dim;
                update_weights_f32(
                    &mut neuron_weights[offset..offset + weight_dim],
                    input,
                    influence,
                );
            }

            if i % 500 == 0 {
                pb.set_position(i as u64);
            }
        }
        pb.finish_and_clear();
        print_done(&format!("Phase 2: Fine-tuning ({} samples)", fine_sample_size));

        // Phase 3: Final BMU assignment - process in chunks for progress visibility
        let pb = ProgressBar::new(context_vecs.len() as u64);
        pb.set_style(bar_style);
        pb.set_message(format!("Phase 3: Final BMU assignment ({} samples)", format_number(context_vecs.len())));

        let mut word_to_bmus: HashMap<String, Vec<usize>> = HashMap::new();

        // Process in chunks to show progress during parallel computation
        let chunk_size = 10_000usize;
        let mut processed = 0usize;

        for chunk_start in (0..context_vecs.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(context_vecs.len());

            // Prepare chunk inputs
            let chunk_inputs: Vec<(usize, Vec<f32>)> = context_vecs[chunk_start..chunk_end]
                .iter()
                .enumerate()
                .map(|(i, (_, v))| (chunk_start + i, v.clone()))
                .collect();

            // Find BMUs for this chunk in parallel using hierarchical search
            let chunk_bmus = find_all_bmus_parallel_fast(&neuron_weights, &chunk_inputs, dim, weight_dim);

            // Record BMUs
            for (sample_idx, bmu_idx) in chunk_bmus {
                let center = &context_vecs[sample_idx].0;
                word_to_bmus
                    .entry(center.clone())
                    .or_default()
                    .push(bmu_idx);
            }

            processed = chunk_end;
            pb.set_position(processed as u64);
        }

        pb.finish_and_clear();
        print_done(&format!("Phase 3: Mapped {} unique words", format_number(word_to_bmus.len())));

        // Copy f32 weights back to neurons as f64
        for (i, neuron) in som.neurons.iter_mut().enumerate() {
            let offset = i * weight_dim;
            for (j, w) in neuron.weights.iter_mut().enumerate() {
                *w = neuron_weights[offset + j] as f64;
            }
        }

        Ok(word_to_bmus)
    }

    /// Original train method for backwards compatibility.
    pub fn train(
        &mut self,
        som: &mut Som,
        contexts: &[TrainingContext],
    ) -> Result<HashMap<String, Vec<usize>>> {
        if contexts.is_empty() {
            return Err(ProteusError::Training(
                "No training contexts provided".to_string(),
            ));
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
            let bmu_idx = som
                .neurons
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
                        if r < 0 || r >= dim as i32 {
                            continue;
                        }
                        r as usize
                    };
                    let nc = if som.toroidal {
                        ((bmu_col as i32 + dc).rem_euclid(dim as i32)) as usize
                    } else {
                        let c = bmu_col as i32 + dc;
                        if c < 0 || c >= dim as i32 {
                            continue;
                        }
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
            (
                "hello".to_string(),
                vec!["world".to_string(), "there".to_string()],
            ),
            (
                "world".to_string(),
                vec!["hello".to_string(), "peace".to_string()],
            ),
        ];

        let embeddings = WordEmbeddings::from_contexts(&contexts, 50, Some(42));

        assert!(embeddings.get("hello").is_some());
        assert!(embeddings.get("world").is_some());
        assert_eq!(embeddings.dim(), 50);
    }
}
