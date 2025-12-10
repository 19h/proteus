//! Text normalization for preprocessing.

use crate::config::TextConfig;
use unicode_normalization::UnicodeNormalization;

/// Text normalizer that applies various transformations.
#[derive(Debug, Clone)]
pub struct Normalizer {
    config: TextConfig,
}

impl Normalizer {
    /// Creates a new normalizer with the given configuration.
    pub fn new(config: TextConfig) -> Self {
        Self { config }
    }

    /// Creates a normalizer with default configuration.
    pub fn default_config() -> Self {
        Self::new(TextConfig::default())
    }

    /// Normalizes a single token.
    ///
    /// Returns `None` if the token should be filtered out.
    pub fn normalize_token(&self, token: &str) -> Option<String> {
        let mut result = token.to_string();

        // Apply Unicode normalization (NFD)
        if self.config.unicode_normalize {
            result = result.nfd().collect();
        }

        // Lowercase
        if self.config.lowercase {
            result = result.to_lowercase();
        }

        // Remove punctuation
        if self.config.remove_punctuation {
            result = result.chars().filter(|c| !c.is_ascii_punctuation()).collect();
        }

        // Check if numeric
        if self.config.remove_numbers && result.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }

        // Check length constraints
        if result.len() < self.config.min_token_length
            || result.len() > self.config.max_token_length
        {
            return None;
        }

        // Filter out empty tokens
        if result.is_empty() {
            return None;
        }

        Some(result)
    }

    /// Normalizes text and returns all valid tokens.
    pub fn normalize_text(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .filter_map(|token| self.normalize_token(token))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lowercase() {
        let normalizer = Normalizer::default_config();
        assert_eq!(normalizer.normalize_token("HELLO"), Some("hello".to_string()));
    }

    #[test]
    fn test_remove_punctuation() {
        let normalizer = Normalizer::default_config();
        assert_eq!(normalizer.normalize_token("hello,"), Some("hello".to_string()));
        assert_eq!(normalizer.normalize_token("world!"), Some("world".to_string()));
    }

    #[test]
    fn test_min_length_filter() {
        let normalizer = Normalizer::default_config();
        assert_eq!(normalizer.normalize_token("a"), None);
        assert_eq!(normalizer.normalize_token("ab"), Some("ab".to_string()));
    }

    #[test]
    fn test_unicode_normalization() {
        let normalizer = Normalizer::default_config();
        // café with combining acute accent vs precomposed
        let result = normalizer.normalize_token("café");
        assert!(result.is_some());
    }

    #[test]
    fn test_normalize_text() {
        let normalizer = Normalizer::default_config();
        let tokens = normalizer.normalize_text("Hello, World! This is a test.");
        assert_eq!(tokens, vec!["hello", "world", "this", "is", "test"]);
    }

    #[test]
    fn test_remove_numbers() {
        let mut config = TextConfig::default();
        config.remove_numbers = true;
        let normalizer = Normalizer::new(config);
        assert_eq!(normalizer.normalize_token("123"), None);
        assert_eq!(normalizer.normalize_token("abc"), Some("abc".to_string()));
        // Mixed alphanumeric should still pass
        assert_eq!(normalizer.normalize_token("abc123"), Some("abc123".to_string()));
    }
}
