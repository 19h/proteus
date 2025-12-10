//! Sentence segmentation using wtpsplit SaT models.
//!
//! This module provides sentence segmentation capabilities that can be
//! integrated with the tokenizer to ensure context windows don't span
//! sentence boundaries.

use crate::wtpsplit::{SaT, SaTOptions};

/// A sentence segmenter using wtpsplit SaT models.
pub struct SentenceSegmenter {
    sat: SaT,
    options: SaTOptions,
}

impl SentenceSegmenter {
    /// Create a new sentence segmenter with a pre-loaded SaT model.
    ///
    /// # Arguments
    /// * `model_name_or_path` - Either a model name (e.g., "sat-3l-sm") or local path
    ///
    /// # Example
    /// ```no_run
    /// use proteus::text::SentenceSegmenter;
    ///
    /// let mut segmenter = SentenceSegmenter::new("sat-3l-sm")?;
    /// let sentences = segmenter.segment("Hello world. This is a test.")?;
    /// # Ok::<(), proteus::wtpsplit::Error>(())
    /// ```
    pub fn new(model_name_or_path: &str) -> crate::wtpsplit::Result<Self> {
        let sat = SaT::new(model_name_or_path, None)?;
        Ok(Self {
            sat,
            options: SaTOptions::default(),
        })
    }

    /// Create a segmenter with custom options.
    pub fn with_options(model_name_or_path: &str, options: SaTOptions) -> crate::wtpsplit::Result<Self> {
        let sat = SaT::new(model_name_or_path, None)?;
        Ok(Self { sat, options })
    }

    /// Set the probability threshold for sentence boundaries.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.options.threshold = Some(threshold);
    }

    /// Set whether to strip whitespace from sentences.
    pub fn set_strip_whitespace(&mut self, strip: bool) {
        self.options.strip_whitespace = strip;
    }

    /// Segment text into sentences.
    ///
    /// # Arguments
    /// * `text` - Input text to segment
    ///
    /// # Returns
    /// A vector of sentence strings
    pub fn segment(&mut self, text: &str) -> crate::wtpsplit::Result<Vec<String>> {
        self.sat.split(text, Some(&self.options))
    }

    /// Segment multiple texts into sentences.
    ///
    /// # Arguments
    /// * `texts` - Slice of texts to segment
    ///
    /// # Returns
    /// A vector of vectors, where each inner vector contains sentences for one input text
    pub fn segment_batch(&mut self, texts: &[&str]) -> crate::wtpsplit::Result<Vec<Vec<String>>> {
        self.sat.split_batch(texts, Some(&self.options))
    }

    /// Get sentence boundary probabilities for text.
    ///
    /// # Arguments
    /// * `text` - Input text
    ///
    /// # Returns
    /// Per-character probabilities of being a sentence boundary
    pub fn predict_proba(&mut self, text: &str) -> crate::wtpsplit::Result<Vec<f32>> {
        self.sat.predict_proba(text, Some(&self.options))
    }
}

#[cfg(test)]
mod tests {
    // Note: Tests require downloading models, so they're marked as ignored by default
    // Run with: cargo test --features=download-models -- --ignored

    #[test]
    #[ignore]
    fn test_segmenter_basic() {
        use super::*;

        let mut segmenter = SentenceSegmenter::new("sat-3l-sm").expect("Failed to load model");
        let sentences = segmenter.segment("Hello world. This is a test.").expect("Segmentation failed");

        assert!(sentences.len() >= 2, "Expected at least 2 sentences, got {}", sentences.len());
    }
}
