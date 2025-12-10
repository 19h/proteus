//! Configuration for the Proteus Semantic Folding engine.

use serde::{Deserialize, Serialize};

/// Main configuration for the Proteus engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// SOM (Self-Organizing Map) configuration.
    pub som: SomConfig,

    /// Text processing configuration.
    pub text: TextConfig,

    /// Fingerprint configuration.
    pub fingerprint: FingerprintConfig,

    /// Storage configuration.
    pub storage: StorageConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            som: SomConfig::default(),
            text: TextConfig::default(),
            fingerprint: FingerprintConfig::default(),
            storage: StorageConfig::default(),
        }
    }
}

/// Self-Organizing Map configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomConfig {
    /// Grid dimension (grid is dim x dim).
    /// Default: 128 (for 16,384 total positions).
    pub dimension: usize,

    /// Number of training iterations.
    /// Default: 100,000.
    pub iterations: usize,

    /// Initial learning rate.
    /// Default: 0.1.
    pub initial_learning_rate: f64,

    /// Final learning rate.
    /// Default: 0.01.
    pub final_learning_rate: f64,

    /// Initial neighborhood radius.
    /// Default: 64 (half of dimension).
    pub initial_radius: f64,

    /// Final neighborhood radius.
    /// Default: 1.0.
    pub final_radius: f64,

    /// Dimensionality of the weight vectors.
    /// This is typically the size of the context embedding.
    /// Default: 300.
    pub weight_dimension: usize,

    /// Context window size for training.
    /// Default: 5.
    pub context_window: usize,

    /// Use toroidal boundary conditions (wrapping edges).
    /// Default: true.
    pub toroidal: bool,

    /// Random seed for reproducibility.
    /// Default: None (random).
    pub seed: Option<u64>,

    /// Number of parallel threads for training.
    /// Default: 0 (use all available cores).
    pub num_threads: usize,
}

impl Default for SomConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            iterations: 100_000,
            initial_learning_rate: 0.1,
            final_learning_rate: 0.01,
            initial_radius: 64.0,
            final_radius: 1.0,
            weight_dimension: 300,
            context_window: 5,
            toroidal: true,
            seed: None,
            num_threads: 0,
        }
    }
}

impl SomConfig {
    /// Returns the total number of neurons in the SOM.
    #[inline]
    pub fn total_neurons(&self) -> usize {
        self.dimension * self.dimension
    }
}

/// Text processing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextConfig {
    /// Convert all text to lowercase.
    /// Default: true.
    pub lowercase: bool,

    /// Minimum token length to include.
    /// Default: 2.
    pub min_token_length: usize,

    /// Maximum token length to include.
    /// Default: 50.
    pub max_token_length: usize,

    /// Remove punctuation from tokens.
    /// Default: true.
    pub remove_punctuation: bool,

    /// Remove numeric tokens.
    /// Default: false.
    pub remove_numbers: bool,

    /// Apply Unicode normalization (NFD).
    /// Default: true.
    pub unicode_normalize: bool,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            min_token_length: 2,
            max_token_length: 50,
            remove_punctuation: true,
            remove_numbers: false,
            unicode_normalize: true,
        }
    }
}

/// Fingerprint configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerprintConfig {
    /// Target sparsity (fraction of active bits).
    /// Default: 0.02 (2%).
    pub sparsity: f64,

    /// Maximum number of active bits in a fingerprint.
    /// Default: 328 (2% of 16,384).
    pub max_active_bits: usize,

    /// Minimum number of active bits in a fingerprint.
    /// Default: 10.
    pub min_active_bits: usize,

    /// Use weighted aggregation for text fingerprints.
    /// Default: true.
    pub weighted_aggregation: bool,
}

impl Default for FingerprintConfig {
    fn default() -> Self {
        Self {
            sparsity: 0.02,
            max_active_bits: 328,
            min_active_bits: 10,
            weighted_aggregation: true,
        }
    }
}

/// Storage configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Use memory-mapped files for large retinas.
    /// Default: true.
    pub use_mmap: bool,

    /// Compression level for binary storage (0-9).
    /// 0 = no compression, 9 = maximum compression.
    /// Default: 0 (no compression for speed).
    pub compression_level: u8,

    /// Maximum cache size for loaded fingerprints.
    /// Default: 100,000.
    pub cache_size: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            use_mmap: true,
            compression_level: 0,
            cache_size: 100_000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.som.dimension, 128);
        assert_eq!(config.som.total_neurons(), 16_384);
        assert_eq!(config.fingerprint.sparsity, 0.02);
    }

    #[test]
    fn test_som_total_neurons() {
        let mut config = SomConfig::default();
        config.dimension = 64;
        assert_eq!(config.total_neurons(), 4096);
    }
}
