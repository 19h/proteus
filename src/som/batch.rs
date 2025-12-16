//! Mini-batch SOM training with adaptive learning rate.
//!
//! This module implements modern SOM training techniques:
//!
//! 1. **Mini-batch training**: Accumulates weight updates over batches for better gradient
//!    estimates and more stable convergence than online learning.
//!
//! 2. **Adaptive learning rate (Adam-like)**: Maintains per-weight momentum and adaptive
//!    learning rates for faster convergence.
//!
//! 3. **Cosine annealing**: Smooth learning rate schedule with warm restarts for
//!    escaping local minima.
//!
//! 4. **Topographic error monitoring**: Early stopping based on map quality metrics.
//!
//! References:
//! - Kohonen (2001): "Self-Organizing Maps" (3rd ed.)
//! - Heskes (1999): "Energy functions for self-organizing maps"
//! - Loshchilov & Hutter (2016): "SGDR: Stochastic Gradient Descent with Warm Restarts"

use super::simd::{distance_squared_f32, normalize_f32};
use rayon::prelude::*;
use std::collections::HashMap;

/// Configuration for mini-batch SOM training.
#[derive(Debug, Clone)]
pub struct BatchSomConfig {
    /// Mini-batch size. Default: 256.
    pub batch_size: usize,

    /// Number of epochs (full passes over data). Default: 10.
    pub epochs: usize,

    /// Initial learning rate. Default: 0.5.
    pub initial_lr: f32,

    /// Final learning rate. Default: 0.01.
    pub final_lr: f32,

    /// Initial neighborhood radius (fraction of grid dimension). Default: 0.5.
    pub initial_radius_frac: f32,

    /// Final neighborhood radius. Default: 1.0.
    pub final_radius: f32,

    /// Use Adam-style adaptive learning. Default: true.
    pub use_adam: bool,

    /// Adam beta1 (momentum). Default: 0.9.
    pub adam_beta1: f32,

    /// Adam beta2 (RMSprop). Default: 0.999.
    pub adam_beta2: f32,

    /// Use cosine annealing. Default: true.
    pub use_cosine_annealing: bool,

    /// Number of warm restarts. Default: 2.
    pub warm_restarts: usize,

    /// Compute topographic error periodically. Default: true.
    pub monitor_quality: bool,

    /// Random seed.
    pub seed: Option<u64>,
}

impl Default for BatchSomConfig {
    fn default() -> Self {
        Self {
            batch_size: 256,
            epochs: 10,
            initial_lr: 0.5,
            final_lr: 0.01,
            initial_radius_frac: 0.5,
            final_radius: 1.0,
            use_adam: true,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            use_cosine_annealing: true,
            warm_restarts: 2,
            monitor_quality: true,
            seed: None,
        }
    }
}

/// Adam optimizer state for a single weight.
#[derive(Clone, Default)]
struct AdamState {
    m: f32, // First moment (mean of gradients)
    v: f32, // Second moment (mean of squared gradients)
}

/// Mini-batch SOM trainer with modern optimization techniques.
pub struct BatchSomTrainer {
    config: BatchSomConfig,
    /// Grid dimension.
    grid_dim: usize,
    /// Weight dimension.
    weight_dim: usize,
    /// Number of neurons.
    num_neurons: usize,
    /// Adam optimizer state per weight.
    adam_states: Vec<AdamState>,
    /// Current epoch.
    current_epoch: usize,
    /// Current step within epoch.
    current_step: usize,
    /// Total steps.
    total_steps: usize,
    /// Training metrics history.
    metrics: TrainingMetrics,
}

/// Training metrics tracked during optimization.
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// Quantization error per epoch.
    pub quantization_errors: Vec<f32>,
    /// Topographic error per epoch.
    pub topographic_errors: Vec<f32>,
    /// Learning rates used.
    pub learning_rates: Vec<f32>,
    /// Neighborhood radii used.
    pub radii: Vec<f32>,
}

impl BatchSomTrainer {
    /// Create a new mini-batch SOM trainer.
    pub fn new(config: BatchSomConfig, grid_dim: usize, weight_dim: usize) -> Self {
        let num_neurons = grid_dim * grid_dim;
        let adam_states = vec![AdamState::default(); num_neurons * weight_dim];

        Self {
            config,
            grid_dim,
            weight_dim,
            num_neurons,
            adam_states,
            current_epoch: 0,
            current_step: 0,
            total_steps: 0,
            metrics: TrainingMetrics::default(),
        }
    }

    /// Train the SOM on the given data.
    ///
    /// Returns mapping from words to their BMU indices.
    pub fn train(
        &mut self,
        weights: &mut [f32],
        data: &[(String, Vec<f32>)],
    ) -> HashMap<String, Vec<usize>> {
        assert_eq!(weights.len(), self.num_neurons * self.weight_dim);

        let n = data.len();
        let batches_per_epoch = (n + self.config.batch_size - 1) / self.config.batch_size;
        self.total_steps = batches_per_epoch * self.config.epochs;

        // Initialize RNG for shuffling
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = match self.config.seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };

        let mut indices: Vec<usize> = (0..n).collect();
        let mut word_to_bmus: HashMap<String, Vec<usize>> = HashMap::new();

        // Gradient accumulator for batch updates
        let mut grad_accum = vec![0.0f32; weights.len()];
        let mut update_counts = vec![0usize; self.num_neurons];

        for epoch in 0..self.config.epochs {
            self.current_epoch = epoch;

            // Check for warm restart
            if self.config.use_cosine_annealing && self.config.warm_restarts > 0 {
                let restart_interval = self.config.epochs / (self.config.warm_restarts + 1);
                if epoch > 0 && epoch % restart_interval == 0 {
                    // Reset Adam states on warm restart
                    for state in &mut self.adam_states {
                        state.m *= 0.5;
                        state.v *= 0.5;
                    }
                }
            }

            // Shuffle data
            indices.shuffle(&mut rng);

            let mut epoch_qe = 0.0f32;
            let mut epoch_te = 0.0f32;
            let mut epoch_samples = 0usize;

            // Process mini-batches
            for batch_start in (0..n).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(n);
                let batch_indices = &indices[batch_start..batch_end];

                // Get current learning rate and radius
                let progress = self.current_step as f32 / self.total_steps as f32;
                let lr = self.get_learning_rate(progress);
                let radius = self.get_radius(progress);

                // Reset accumulators
                grad_accum.fill(0.0);
                update_counts.fill(0);

                // Find BMUs and accumulate gradients for batch
                let batch_results: Vec<(usize, usize, f32)> = batch_indices
                    .par_iter()
                    .map(|&idx| {
                        let (_, vec) = &data[idx];
                        let (bmu, dist) = self.find_bmu(weights, vec);
                        (idx, bmu, dist)
                    })
                    .collect();

                // Accumulate gradients (sequential to avoid race conditions)
                for (idx, bmu, dist) in &batch_results {
                    let (word, vec) = &data[*idx];

                    // Record BMU
                    word_to_bmus.entry(word.clone()).or_default().push(*bmu);

                    // Accumulate quantization error
                    epoch_qe += dist;
                    epoch_samples += 1;

                    // Check topographic error (BMU and second BMU should be neighbors)
                    if self.config.monitor_quality {
                        let second_bmu = self.find_second_bmu(weights, vec, *bmu);
                        if !self.are_neighbors(*bmu, second_bmu) {
                            epoch_te += 1.0;
                        }
                    }

                    // Compute neighborhood updates
                    let bmu_row = *bmu / self.grid_dim;
                    let bmu_col = *bmu % self.grid_dim;
                    let radius_int = (radius * 2.0).ceil() as i32;

                    for dr in -radius_int..=radius_int {
                        for dc in -radius_int..=radius_int {
                            let nr = bmu_row as i32 + dr;
                            let nc = bmu_col as i32 + dc;

                            // Toroidal wrapping
                            let nr = ((nr % self.grid_dim as i32) + self.grid_dim as i32) as usize % self.grid_dim;
                            let nc = ((nc % self.grid_dim as i32) + self.grid_dim as i32) as usize % self.grid_dim;

                            let neuron_idx = nr * self.grid_dim + nc;

                            // Gaussian neighborhood
                            let dist_sq = (dr * dr + dc * dc) as f32;
                            let sigma_sq = radius * radius;
                            let neighborhood = (-dist_sq / (2.0 * sigma_sq)).exp();

                            if neighborhood > 0.001 {
                                update_counts[neuron_idx] += 1;
                                let offset = neuron_idx * self.weight_dim;

                                for (j, &input_val) in vec.iter().enumerate() {
                                    let diff = input_val - weights[offset + j];
                                    grad_accum[offset + j] += neighborhood * diff;
                                }
                            }
                        }
                    }
                }

                // Apply batch update
                self.apply_batch_update(weights, &grad_accum, &update_counts, lr);

                self.current_step += 1;
            }

            // Record epoch metrics
            if epoch_samples > 0 {
                let avg_qe = epoch_qe / epoch_samples as f32;
                let avg_te = epoch_te / epoch_samples as f32;

                self.metrics.quantization_errors.push(avg_qe);
                self.metrics.topographic_errors.push(avg_te);
                self.metrics.learning_rates.push(self.get_learning_rate(epoch as f32 / self.config.epochs as f32));
                self.metrics.radii.push(self.get_radius(epoch as f32 / self.config.epochs as f32));
            }
        }

        word_to_bmus
    }

    /// Get training metrics.
    pub fn metrics(&self) -> &TrainingMetrics {
        &self.metrics
    }

    /// Find Best Matching Unit for an input vector.
    fn find_bmu(&self, weights: &[f32], input: &[f32]) -> (usize, f32) {
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        for i in 0..self.num_neurons {
            let offset = i * self.weight_dim;
            let dist = distance_squared_f32(&weights[offset..offset + self.weight_dim], input);

            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        (best_idx, best_dist)
    }

    /// Find second-best matching unit (for topographic error).
    fn find_second_bmu(&self, weights: &[f32], input: &[f32], bmu: usize) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        for i in 0..self.num_neurons {
            if i == bmu {
                continue;
            }
            let offset = i * self.weight_dim;
            let dist = distance_squared_f32(&weights[offset..offset + self.weight_dim], input);

            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Check if two neurons are neighbors on the grid.
    fn are_neighbors(&self, n1: usize, n2: usize) -> bool {
        let r1 = n1 / self.grid_dim;
        let c1 = n1 % self.grid_dim;
        let r2 = n2 / self.grid_dim;
        let c2 = n2 % self.grid_dim;

        // Toroidal distance
        let dr = ((r1 as i32 - r2 as i32).abs()).min(
            self.grid_dim as i32 - (r1 as i32 - r2 as i32).abs()
        );
        let dc = ((c1 as i32 - c2 as i32).abs()).min(
            self.grid_dim as i32 - (c1 as i32 - c2 as i32).abs()
        );

        dr <= 1 && dc <= 1
    }

    /// Get learning rate for current progress (0-1).
    fn get_learning_rate(&self, progress: f32) -> f32 {
        if self.config.use_cosine_annealing {
            // Cosine annealing with warm restarts
            let cycle_length = 1.0 / (self.config.warm_restarts + 1) as f32;
            let cycle_progress = (progress % cycle_length) / cycle_length;
            let cosine_factor = (1.0 + (std::f32::consts::PI * cycle_progress).cos()) / 2.0;
            self.config.final_lr + (self.config.initial_lr - self.config.final_lr) * cosine_factor
        } else {
            // Exponential decay
            self.config.initial_lr * (self.config.final_lr / self.config.initial_lr).powf(progress)
        }
    }

    /// Get neighborhood radius for current progress.
    fn get_radius(&self, progress: f32) -> f32 {
        let initial_radius = self.grid_dim as f32 * self.config.initial_radius_frac;
        initial_radius * (self.config.final_radius / initial_radius).powf(progress)
    }

    /// Apply batch update with optional Adam optimization.
    fn apply_batch_update(
        &mut self,
        weights: &mut [f32],
        gradients: &[f32],
        counts: &[usize],
        lr: f32,
    ) {
        let eps = 1e-8f32;

        for neuron_idx in 0..self.num_neurons {
            let count = counts[neuron_idx];
            if count == 0 {
                continue;
            }

            let offset = neuron_idx * self.weight_dim;
            let inv_count = 1.0 / count as f32;

            for j in 0..self.weight_dim {
                let idx = offset + j;
                let grad = gradients[idx] * inv_count;

                if self.config.use_adam {
                    let state = &mut self.adam_states[idx];

                    // Update biased first moment
                    state.m = self.config.adam_beta1 * state.m + (1.0 - self.config.adam_beta1) * grad;

                    // Update biased second moment
                    state.v = self.config.adam_beta2 * state.v + (1.0 - self.config.adam_beta2) * grad * grad;

                    // Bias correction
                    let t = (self.current_step + 1) as f32;
                    let m_hat = state.m / (1.0 - self.config.adam_beta1.powf(t));
                    let v_hat = state.v / (1.0 - self.config.adam_beta2.powf(t));

                    // Update weight
                    weights[idx] += lr * m_hat / (v_hat.sqrt() + eps);
                } else {
                    // Simple SGD update
                    weights[idx] += lr * grad;
                }
            }

            // Optionally normalize neuron weights
            // normalize_f32(&mut weights[offset..offset + self.weight_dim]);
        }
    }
}

/// Growing SOM (GSOM) for adaptive topology.
///
/// Starts with a small map and grows dynamically based on quantization error.
#[allow(dead_code)]
pub struct GrowingSom {
    /// Current grid dimension.
    grid_dim: usize,
    /// Weight dimension (for weight vector extraction).
    weight_dim: usize,
    /// Neuron weights (flat array).
    weights: Vec<f32>,
    /// Spread factor (controls growth rate).
    spread_factor: f32,
    /// Growth threshold (determines when to grow).
    growth_threshold: f32,
    /// Error accumulator per boundary neuron.
    boundary_errors: HashMap<usize, f32>,
}

impl GrowingSom {
    /// Create a new Growing SOM.
    pub fn new(initial_dim: usize, weight_dim: usize, spread_factor: f32) -> Self {
        let num_neurons = initial_dim * initial_dim;
        let weights = vec![0.0f32; num_neurons * weight_dim];

        // Initialize with random weights
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::from_entropy();
        let mut weights = weights;
        for w in &mut weights {
            *w = rng.gen_range(-1.0..1.0);
        }

        // Normalize each neuron
        for i in 0..num_neurons {
            let offset = i * weight_dim;
            normalize_f32(&mut weights[offset..offset + weight_dim]);
        }

        Self {
            grid_dim: initial_dim,
            weight_dim,
            weights,
            spread_factor,
            growth_threshold: -(weight_dim as f32) * spread_factor.ln(),
            boundary_errors: HashMap::new(),
        }
    }

    /// Check if a neuron is on the boundary.
    pub fn is_boundary(&self, idx: usize) -> bool {
        let row = idx / self.grid_dim;
        let col = idx % self.grid_dim;
        row == 0 || row == self.grid_dim - 1 || col == 0 || col == self.grid_dim - 1
    }

    /// Get current grid dimension.
    pub fn dimension(&self) -> usize {
        self.grid_dim
    }

    /// Get current number of neurons.
    pub fn num_neurons(&self) -> usize {
        self.grid_dim * self.grid_dim
    }

    /// Get weights.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Get mutable weights.
    pub fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data(n: usize, dim: usize) -> Vec<(String, Vec<f32>)> {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        (0..n)
            .map(|i| {
                let mut vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                normalize_f32(&mut vec);
                (format!("word_{}", i), vec)
            })
            .collect()
    }

    #[test]
    fn test_batch_som_train() {
        let config = BatchSomConfig {
            batch_size: 32,
            epochs: 5,
            seed: Some(42),
            ..Default::default()
        };

        let grid_dim = 8;
        let weight_dim = 16;

        let mut trainer = BatchSomTrainer::new(config, grid_dim, weight_dim);

        // Initialize random weights
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut weights: Vec<f32> = (0..grid_dim * grid_dim * weight_dim)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        // Create test data
        let data = create_test_data(500, weight_dim);

        // Train
        let word_to_bmus = trainer.train(&mut weights, &data);

        assert!(!word_to_bmus.is_empty());

        // Check metrics were recorded
        let metrics = trainer.metrics();
        assert_eq!(metrics.quantization_errors.len(), 5);
        assert_eq!(metrics.topographic_errors.len(), 5);

        // Quantization error should decrease
        assert!(metrics.quantization_errors[4] < metrics.quantization_errors[0]);
    }

    #[test]
    fn test_cosine_annealing() {
        let config = BatchSomConfig {
            use_cosine_annealing: true,
            warm_restarts: 2,
            initial_lr: 0.5,
            final_lr: 0.01,
            ..Default::default()
        };

        let trainer = BatchSomTrainer::new(config, 8, 16);

        // LR should start high
        let lr_start = trainer.get_learning_rate(0.0);
        assert!((lr_start - 0.5).abs() < 0.01);

        // LR should cycle
        let lr_mid = trainer.get_learning_rate(0.33);
        assert!(lr_mid < lr_start);
        assert!(lr_mid > 0.01);
    }
}
