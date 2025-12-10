//! Neuron representation for the Self-Organizing Map.

use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// A neuron in the Self-Organizing Map.
///
/// Each neuron has a position on the 2D grid and a weight vector
/// that represents its learned semantic context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuron {
    /// Row position on the grid.
    pub row: usize,
    /// Column position on the grid.
    pub col: usize,
    /// Weight vector representing the neuron's learned context.
    pub weights: Vec<f64>,
}

impl Neuron {
    /// Creates a new neuron with random weights.
    ///
    /// Weights are initialized from a normal distribution with mean 0 and std 0.1.
    pub fn new_random<R: Rng>(row: usize, col: usize, weight_dim: usize, rng: &mut R) -> Self {
        let normal = Normal::new(0.0, 0.1).unwrap();
        let weights: Vec<f64> = (0..weight_dim).map(|_| normal.sample(rng)).collect();

        Self { row, col, weights }
    }

    /// Creates a new neuron with zero weights.
    pub fn new_zeros(row: usize, col: usize, weight_dim: usize) -> Self {
        Self {
            row,
            col,
            weights: vec![0.0; weight_dim],
        }
    }

    /// Creates a new neuron with the given weights.
    pub fn new_with_weights(row: usize, col: usize, weights: Vec<f64>) -> Self {
        Self { row, col, weights }
    }

    /// Returns the 1D index for this neuron in a grid of the given dimension.
    #[inline]
    pub fn index(&self, dimension: usize) -> usize {
        self.row * dimension + self.col
    }

    /// Computes the Euclidean distance between this neuron's weights and an input vector.
    pub fn distance(&self, input: &[f64]) -> f64 {
        debug_assert_eq!(
            self.weights.len(),
            input.len(),
            "Weight and input dimensions must match"
        );

        self.weights
            .iter()
            .zip(input.iter())
            .map(|(w, i)| (w - i).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Computes the squared Euclidean distance (faster, avoids sqrt).
    #[inline]
    pub fn distance_squared(&self, input: &[f64]) -> f64 {
        self.weights
            .iter()
            .zip(input.iter())
            .map(|(w, i)| (w - i).powi(2))
            .sum()
    }

    /// Computes the grid distance to another neuron.
    ///
    /// If `toroidal` is true, uses toroidal (wrapping) distance.
    pub fn grid_distance(&self, other: &Neuron, dimension: usize, toroidal: bool) -> f64 {
        let (dr, dc) = if toroidal {
            let dr = self.row as i32 - other.row as i32;
            let dc = self.col as i32 - other.col as i32;
            let dim = dimension as i32;

            let dr = dr.abs().min(dim - dr.abs());
            let dc = dc.abs().min(dim - dc.abs());
            (dr as f64, dc as f64)
        } else {
            let dr = (self.row as f64 - other.row as f64).abs();
            let dc = (self.col as f64 - other.col as f64).abs();
            (dr, dc)
        };

        (dr * dr + dc * dc).sqrt()
    }

    /// Computes the squared grid distance (faster, avoids sqrt).
    #[inline]
    pub fn grid_distance_squared(&self, other: &Neuron, dimension: usize, toroidal: bool) -> f64 {
        let dist = self.grid_distance(other, dimension, toroidal);
        dist * dist
    }

    /// Updates the neuron's weights towards an input vector.
    ///
    /// `learning_rate` is the overall learning rate.
    /// `neighborhood` is the neighborhood influence (0.0 to 1.0).
    pub fn update_weights(&mut self, input: &[f64], learning_rate: f64, neighborhood: f64) {
        let influence = learning_rate * neighborhood;

        for (w, i) in self.weights.iter_mut().zip(input.iter()) {
            *w += influence * (i - *w);
        }
    }

    /// Normalizes the weight vector to unit length.
    pub fn normalize_weights(&mut self) {
        let norm: f64 = self.weights.iter().map(|w| w * w).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for w in &mut self.weights {
                *w /= norm;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_neuron_creation() {
        let neuron = Neuron::new_zeros(5, 10, 100);
        assert_eq!(neuron.row, 5);
        assert_eq!(neuron.col, 10);
        assert_eq!(neuron.weights.len(), 100);
        assert!(neuron.weights.iter().all(|&w| w == 0.0));
    }

    #[test]
    fn test_random_initialization() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let neuron = Neuron::new_random(0, 0, 100, &mut rng);
        assert_eq!(neuron.weights.len(), 100);
        // Weights should not all be zero
        assert!(neuron.weights.iter().any(|&w| w != 0.0));
    }

    #[test]
    fn test_index() {
        let neuron = Neuron::new_zeros(5, 10, 100);
        assert_eq!(neuron.index(128), 5 * 128 + 10);
    }

    #[test]
    fn test_distance() {
        let neuron = Neuron::new_with_weights(0, 0, vec![1.0, 0.0, 0.0]);
        let input = vec![0.0, 1.0, 0.0];
        let dist = neuron.distance(&input);
        assert!((dist - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_grid_distance_non_toroidal() {
        let n1 = Neuron::new_zeros(0, 0, 10);
        let n2 = Neuron::new_zeros(3, 4, 10);
        let dist = n1.grid_distance(&n2, 128, false);
        assert!((dist - 5.0).abs() < 1e-10); // 3-4-5 triangle
    }

    #[test]
    fn test_grid_distance_toroidal() {
        let n1 = Neuron::new_zeros(0, 0, 10);
        let n2 = Neuron::new_zeros(127, 127, 10);
        // With toroidal distance, these should be close (distance 1,1)
        let dist = n1.grid_distance(&n2, 128, true);
        assert!((dist - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_update_weights() {
        let mut neuron = Neuron::new_with_weights(0, 0, vec![0.0, 0.0, 0.0]);
        let input = vec![1.0, 1.0, 1.0];
        neuron.update_weights(&input, 0.5, 1.0);
        assert!((neuron.weights[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_weights() {
        let mut neuron = Neuron::new_with_weights(0, 0, vec![3.0, 4.0, 0.0]);
        neuron.normalize_weights();
        let norm: f64 = neuron.weights.iter().map(|w| w * w).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }
}
