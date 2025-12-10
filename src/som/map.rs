//! Self-Organizing Map (SOM) implementation.

use crate::config::SomConfig;
use crate::error::{ProteusError, Result};
use crate::som::Neuron;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A Self-Organizing Map for learning semantic space representations.
///
/// The SOM is a 2D grid of neurons, where each neuron has a weight vector
/// representing a semantic context. Through training on word co-occurrence
/// patterns, the map learns to organize semantically similar contexts
/// in nearby grid positions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Som {
    /// Grid dimension (grid is dimension x dimension).
    pub dimension: usize,
    /// The neurons in the grid (row-major order).
    pub neurons: Vec<Neuron>,
    /// Weight vector dimensionality.
    pub weight_dim: usize,
    /// Use toroidal boundary conditions.
    pub toroidal: bool,
}

impl Som {
    /// Creates a new SOM with randomly initialized weights.
    pub fn new(config: &SomConfig) -> Self {
        let dimension = config.dimension;
        let weight_dim = config.weight_dimension;
        let total = dimension * dimension;

        let mut rng = match config.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::from_entropy(),
        };

        let neurons: Vec<Neuron> = (0..total)
            .map(|i| {
                let row = i / dimension;
                let col = i % dimension;
                Neuron::new_random(row, col, weight_dim, &mut rng)
            })
            .collect();

        Self {
            dimension,
            neurons,
            weight_dim,
            toroidal: config.toroidal,
        }
    }

    /// Creates a new SOM with zero-initialized weights.
    pub fn new_zeros(dimension: usize, weight_dim: usize, toroidal: bool) -> Self {
        let total = dimension * dimension;
        let neurons: Vec<Neuron> = (0..total)
            .map(|i| {
                let row = i / dimension;
                let col = i % dimension;
                Neuron::new_zeros(row, col, weight_dim)
            })
            .collect();

        Self {
            dimension,
            neurons,
            weight_dim,
            toroidal,
        }
    }

    /// Returns the total number of neurons.
    #[inline]
    pub fn total_neurons(&self) -> usize {
        self.neurons.len()
    }

    /// Gets a neuron by its 1D index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&Neuron> {
        self.neurons.get(index)
    }

    /// Gets a mutable reference to a neuron by its 1D index.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Neuron> {
        self.neurons.get_mut(index)
    }

    /// Gets a neuron by its 2D position.
    #[inline]
    pub fn get_at(&self, row: usize, col: usize) -> Option<&Neuron> {
        if row < self.dimension && col < self.dimension {
            Some(&self.neurons[row * self.dimension + col])
        } else {
            None
        }
    }

    /// Gets a mutable reference to a neuron by its 2D position.
    #[inline]
    pub fn get_at_mut(&mut self, row: usize, col: usize) -> Option<&mut Neuron> {
        if row < self.dimension && col < self.dimension {
            Some(&mut self.neurons[row * self.dimension + col])
        } else {
            None
        }
    }

    /// Finds the Best Matching Unit (BMU) for an input vector.
    ///
    /// The BMU is the neuron whose weight vector is closest to the input.
    /// Returns the index of the BMU.
    pub fn find_bmu(&self, input: &[f64]) -> Result<usize> {
        if input.len() != self.weight_dim {
            return Err(ProteusError::Som(format!(
                "Input dimension {} does not match weight dimension {}",
                input.len(),
                self.weight_dim
            )));
        }

        let (bmu_idx, _) = self
            .neurons
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let dist_a = a.distance_squared(input);
                let dist_b = b.distance_squared(input);
                dist_a.partial_cmp(&dist_b).unwrap()
            })
            .ok_or_else(|| ProteusError::Som("Empty SOM".to_string()))?;

        Ok(bmu_idx)
    }

    /// Finds the Best Matching Unit (BMU) in parallel.
    ///
    /// More efficient for large maps.
    pub fn find_bmu_parallel(&self, input: &[f64]) -> Result<usize> {
        if input.len() != self.weight_dim {
            return Err(ProteusError::Som(format!(
                "Input dimension {} does not match weight dimension {}",
                input.len(),
                self.weight_dim
            )));
        }

        let (bmu_idx, _) = self
            .neurons
            .par_iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let dist_a = a.distance_squared(input);
                let dist_b = b.distance_squared(input);
                dist_a.partial_cmp(&dist_b).unwrap()
            })
            .ok_or_else(|| ProteusError::Som("Empty SOM".to_string()))?;

        Ok(bmu_idx)
    }

    /// Finds the k nearest neurons to an input vector.
    ///
    /// Returns a vector of (index, distance) pairs, sorted by distance.
    pub fn find_k_nearest(&self, input: &[f64], k: usize) -> Result<Vec<(usize, f64)>> {
        if input.len() != self.weight_dim {
            return Err(ProteusError::Som(format!(
                "Input dimension {} does not match weight dimension {}",
                input.len(),
                self.weight_dim
            )));
        }

        let mut distances: Vec<(usize, f64)> = self
            .neurons
            .iter()
            .enumerate()
            .map(|(i, n)| (i, n.distance(input)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);

        Ok(distances)
    }

    /// Computes the neighborhood influence for a neuron relative to the BMU.
    ///
    /// Uses a Gaussian neighborhood function.
    pub fn neighborhood(&self, bmu_idx: usize, neuron_idx: usize, radius: f64) -> f64 {
        let bmu = &self.neurons[bmu_idx];
        let neuron = &self.neurons[neuron_idx];

        let grid_dist = bmu.grid_distance(neuron, self.dimension, self.toroidal);
        let sigma_sq = radius * radius;

        (-grid_dist * grid_dist / (2.0 * sigma_sq)).exp()
    }

    /// Updates all neurons in response to an input.
    ///
    /// `bmu_idx` is the index of the Best Matching Unit.
    /// `learning_rate` is the current learning rate.
    /// `radius` is the current neighborhood radius.
    pub fn update(&mut self, input: &[f64], bmu_idx: usize, learning_rate: f64, radius: f64) {
        let bmu = self.neurons[bmu_idx].clone();

        for neuron in &mut self.neurons {
            let grid_dist = bmu.grid_distance(neuron, self.dimension, self.toroidal);

            // Only update neurons within a reasonable distance (optimization)
            if grid_dist <= radius * 3.0 {
                let sigma_sq = radius * radius;
                let neighborhood = (-grid_dist * grid_dist / (2.0 * sigma_sq)).exp();
                neuron.update_weights(input, learning_rate, neighborhood);
            }
        }
    }

    /// Converts a 1D index to 2D coordinates.
    #[inline]
    pub fn index_to_coords(&self, index: usize) -> (usize, usize) {
        (index / self.dimension, index % self.dimension)
    }

    /// Converts 2D coordinates to a 1D index.
    #[inline]
    pub fn coords_to_index(&self, row: usize, col: usize) -> usize {
        row * self.dimension + col
    }

    /// Normalizes all neuron weights to unit length.
    pub fn normalize_all_weights(&mut self) {
        for neuron in &mut self.neurons {
            neuron.normalize_weights();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SomConfig {
        SomConfig {
            dimension: 8,
            weight_dimension: 10,
            seed: Some(42),
            toroidal: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_som_creation() {
        let config = test_config();
        let som = Som::new(&config);

        assert_eq!(som.dimension, 8);
        assert_eq!(som.total_neurons(), 64);
        assert_eq!(som.weight_dim, 10);
    }

    #[test]
    fn test_neuron_positions() {
        let config = test_config();
        let som = Som::new(&config);

        // Check that neurons have correct positions
        for i in 0..som.total_neurons() {
            let neuron = som.get(i).unwrap();
            let expected_row = i / 8;
            let expected_col = i % 8;
            assert_eq!(neuron.row, expected_row);
            assert_eq!(neuron.col, expected_col);
        }
    }

    #[test]
    fn test_find_bmu() {
        let som = Som::new_zeros(4, 3, false);
        // Set one neuron to have weights close to [1, 0, 0]
        let mut som = som;
        som.neurons[5].weights = vec![1.0, 0.0, 0.0];

        let input = vec![1.0, 0.0, 0.0];
        let bmu_idx = som.find_bmu(&input).unwrap();
        assert_eq!(bmu_idx, 5);
    }

    #[test]
    fn test_find_k_nearest() {
        let mut som = Som::new_zeros(4, 3, false);
        som.neurons[0].weights = vec![1.0, 0.0, 0.0];
        som.neurons[1].weights = vec![0.9, 0.1, 0.0];
        som.neurons[2].weights = vec![0.8, 0.2, 0.0];

        let input = vec![1.0, 0.0, 0.0];
        let nearest = som.find_k_nearest(&input, 3).unwrap();

        assert_eq!(nearest.len(), 3);
        assert_eq!(nearest[0].0, 0);
        assert_eq!(nearest[1].0, 1);
        assert_eq!(nearest[2].0, 2);
    }

    #[test]
    fn test_neighborhood() {
        let config = test_config();
        let som = Som::new(&config);

        // Same neuron should have full neighborhood
        let n = som.neighborhood(0, 0, 2.0);
        assert!((n - 1.0).abs() < 1e-10);

        // Far neuron should have smaller neighborhood
        let n_far = som.neighborhood(0, 63, 2.0);
        assert!(n_far < n);
    }

    #[test]
    fn test_update() {
        let mut som = Som::new_zeros(4, 3, false);
        let input = vec![1.0, 1.0, 1.0];

        som.update(&input, 0, 0.5, 2.0);

        // BMU should have moved towards input
        assert!(som.neurons[0].weights[0] > 0.0);
    }

    #[test]
    fn test_coordinate_conversion() {
        let config = test_config();
        let som = Som::new(&config);

        assert_eq!(som.index_to_coords(10), (1, 2));
        assert_eq!(som.coords_to_index(1, 2), 10);
    }
}
