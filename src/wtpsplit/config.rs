//! Model configuration structures

use serde::{Deserialize, Serialize};

/// Configuration for wtpsplit models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model architecture name
    #[serde(default)]
    pub architectures: Vec<String>,

    /// Model type identifier
    #[serde(default)]
    pub model_type: String,

    /// Number of labels for token classification
    #[serde(default = "default_num_labels")]
    pub num_labels: usize,

    /// Hidden size of the transformer
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    /// Number of attention heads
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,

    /// Number of hidden layers
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,

    /// Maximum position embeddings
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    /// Vocabulary size
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    /// Downsampling rate (for LACanine models)
    #[serde(default = "default_downsampling_rate")]
    pub downsampling_rate: usize,

    /// Number of hash functions (for WtP character models)
    #[serde(default = "default_num_hash_functions")]
    pub num_hash_functions: usize,

    /// Number of hash buckets (for WtP character models)
    #[serde(default = "default_num_hash_buckets")]
    pub num_hash_buckets: usize,

    /// Language adapter mode
    #[serde(default)]
    pub language_adapter: String,

    /// Lookahead configuration
    #[serde(default)]
    pub lookahead: Option<usize>,

    /// Base model name
    #[serde(default)]
    pub base_model: String,
}

fn default_num_labels() -> usize {
    1
}

fn default_hidden_size() -> usize {
    768
}

fn default_num_attention_heads() -> usize {
    12
}

fn default_num_hidden_layers() -> usize {
    12
}

fn default_max_position_embeddings() -> usize {
    514
}

fn default_vocab_size() -> usize {
    250002
}

fn default_downsampling_rate() -> usize {
    1
}

fn default_num_hash_functions() -> usize {
    8
}

fn default_num_hash_buckets() -> usize {
    8192
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architectures: vec![],
            model_type: String::new(),
            num_labels: default_num_labels(),
            hidden_size: default_hidden_size(),
            num_attention_heads: default_num_attention_heads(),
            num_hidden_layers: default_num_hidden_layers(),
            max_position_embeddings: default_max_position_embeddings(),
            vocab_size: default_vocab_size(),
            downsampling_rate: default_downsampling_rate(),
            num_hash_functions: default_num_hash_functions(),
            num_hash_buckets: default_num_hash_buckets(),
            language_adapter: String::new(),
            lookahead: None,
            base_model: String::new(),
        }
    }
}

impl ModelConfig {
    /// Load configuration from a JSON file
    pub fn from_file(path: &std::path::Path) -> crate::wtpsplit::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Check if this is a subword (XLM-RoBERTa based) model
    pub fn is_subword_model(&self) -> bool {
        self.model_type.contains("xlm") || self.base_model.contains("xlm-roberta")
    }

    /// Check if this is a character-level model (WtP)
    pub fn is_char_model(&self) -> bool {
        !self.is_subword_model()
    }
}
