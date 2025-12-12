//! SaT (Segment any Text) - Modern subword-based sentence segmentation
//!
//! SaT models use XLM-RoBERTa as the backbone and operate on subword tokens.
//! They provide better performance and efficiency compared to the legacy WtP models.

use std::path::Path;
use tokenizers::Tokenizer;

use crate::wtpsplit::config::ModelConfig;
use crate::wtpsplit::extract::{extract_sat, logits_to_probs, ExtractConfig, Weighting};
use crate::wtpsplit::hub::{download_model, download_xlm_roberta_tokenizer, get_local_model_files, is_local_path, SAT_HUB_PREFIX};
use crate::wtpsplit::model::OnnxModel;
use crate::wtpsplit::utils::{indices_to_sentences, reinsert_space_probs, remove_spaces};
use crate::wtpsplit::Result;

/// Configuration options for SaT splitting
#[derive(Debug, Clone)]
pub struct SaTOptions {
    /// Probability threshold for sentence boundaries
    pub threshold: Option<f32>,
    /// Stride for overlapping chunks
    pub stride: usize,
    /// Maximum block size
    pub block_size: usize,
    /// Batch size for inference
    pub batch_size: usize,
    /// Whether to pad the last batch
    pub pad_last_batch: bool,
    /// Weighting scheme for overlapping predictions
    pub weighting: Weighting,
    /// Remove whitespace before inference (for some languages)
    pub remove_whitespace_before_inference: bool,
    /// Paragraph threshold (for paragraph segmentation)
    pub paragraph_threshold: f32,
    /// Strip whitespace from sentences
    pub strip_whitespace: bool,
    /// Perform paragraph segmentation
    pub do_paragraph_segmentation: bool,
    /// Split on input newlines
    pub split_on_input_newlines: bool,
}

impl Default for SaTOptions {
    fn default() -> Self {
        Self {
            threshold: None,
            stride: 64,
            block_size: 512,
            batch_size: 32,
            pad_last_batch: false,
            weighting: Weighting::Uniform,
            remove_whitespace_before_inference: false,
            paragraph_threshold: 0.5,
            strip_whitespace: false,
            do_paragraph_segmentation: false,
            split_on_input_newlines: true,
        }
    }
}

/// SaT sentence segmentation model
pub struct SaT {
    model: OnnxModel,
    tokenizer: Tokenizer,
    #[allow(dead_code)]
    model_name: String,
}

impl SaT {
    /// Create a new SaT instance from a model name or path
    ///
    /// # Arguments
    /// * `model_name_or_path` - Either a model name (e.g., "sat-3l-sm") or a local path
    /// * `hub_prefix` - Optional hub prefix (defaults to "segment-any-text")
    ///
    /// # Example
    /// ```no_run
    /// use proteus::wtpsplit::SaT;
    ///
    /// // Load from HuggingFace Hub
    /// let sat = SaT::new("sat-3l-sm", None)?;
    ///
    /// // Load from local path
    /// let sat = SaT::new("/path/to/model", None)?;
    /// # Ok::<(), proteus::wtpsplit::Error>(())
    /// ```
    pub fn new(model_name_or_path: &str, hub_prefix: Option<&str>) -> Result<Self> {
        let hub_prefix = hub_prefix.unwrap_or(SAT_HUB_PREFIX);

        // Get model files (download if necessary)
        let model_files = if is_local_path(model_name_or_path) {
            get_local_model_files(Path::new(model_name_or_path), true)?
        } else {
            download_model(model_name_or_path, Some(hub_prefix), true)?
        };

        // Load config
        let config = ModelConfig::from_file(&model_files.config_path)?;

        // Load ONNX model
        let model = OnnxModel::new(&model_files.onnx_path, config)?;

        // Load tokenizer (use provided one or download XLM-RoBERTa)
        let tokenizer = if let Some(tokenizer_path) = model_files.tokenizer_path {
            Tokenizer::from_file(&tokenizer_path)?
        } else {
            let tokenizer_path = download_xlm_roberta_tokenizer()?;
            Tokenizer::from_file(&tokenizer_path)?
        };

        Ok(Self {
            model,
            tokenizer,
            model_name: model_name_or_path.to_string(),
        })
    }

    /// Load from a local directory
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        Self::new(model_dir.to_str().unwrap_or(""), None)
    }

    /// Get the default threshold based on model name
    ///
    /// The default threshold is 0.01 (same as Python wtpsplit)
    fn get_default_threshold(&self) -> f32 {
        0.01
    }

    /// Get sentence boundary probabilities for a text
    ///
    /// # Arguments
    /// * `text` - Input text
    /// * `options` - Optional configuration options
    ///
    /// # Returns
    /// Per-character probabilities of being a sentence boundary
    pub fn predict_proba(&mut self, text: &str, options: Option<&SaTOptions>) -> Result<Vec<f32>> {
        let options = options.cloned().unwrap_or_default();
        let results = self.predict_proba_batch(&[text], Some(&options))?;
        Ok(results.into_iter().next().unwrap_or_default())
    }

    /// Get sentence boundary probabilities for a batch of texts
    pub fn predict_proba_batch(
        &mut self,
        texts: &[&str],
        options: Option<&SaTOptions>,
    ) -> Result<Vec<Vec<f32>>> {
        let options = options.cloned().unwrap_or_default();

        // Handle whitespace removal if requested
        let (input_texts, space_positions): (Vec<String>, Vec<Vec<usize>>) =
            if options.remove_whitespace_before_inference {
                texts.iter().map(|t| remove_spaces(t)).unzip()
            } else {
                (
                    texts.iter().map(|s| s.to_string()).collect(),
                    vec![vec![]; texts.len()],
                )
            };

        // Filter out empty strings
        let non_empty_indices: Vec<usize> = input_texts
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.trim().is_empty())
            .map(|(i, _)| i)
            .collect();

        let non_empty_texts: Vec<&str> = non_empty_indices
            .iter()
            .map(|&i| input_texts[i].as_str())
            .collect();

        // Run extraction
        let extract_config = ExtractConfig {
            block_size: options.block_size,
            stride: options.stride,
            batch_size: options.batch_size,
            pad_last_batch: options.pad_last_batch,
            weighting: options.weighting,
        };

        let extraction_result = if non_empty_texts.is_empty() {
            crate::wtpsplit::extract::ExtractionResult {
                logits: vec![],
                offset_mappings: None,
            }
        } else {
            extract_sat(&non_empty_texts, &mut self.model, &self.tokenizer, &extract_config)?
        };

        // Convert logits to probabilities
        let mut all_probs: Vec<Vec<f32>> = vec![vec![]; texts.len()];

        for (result_idx, &text_idx) in non_empty_indices.iter().enumerate() {
            let probs = logits_to_probs(&extraction_result.logits[result_idx]);
            all_probs[text_idx] = probs;
        }

        // Fill empty texts with empty probabilities
        for i in 0..texts.len() {
            if all_probs[i].is_empty() && !texts[i].is_empty() {
                all_probs[i] = vec![0.0; texts[i].chars().count()];
            }
        }

        // Reinsert space probabilities if whitespace was removed
        if options.remove_whitespace_before_inference {
            for (probs, positions) in all_probs.iter_mut().zip(space_positions.iter()) {
                if !positions.is_empty() {
                    *probs = reinsert_space_probs(probs, positions);
                }
            }
        }

        Ok(all_probs)
    }

    /// Split text into sentences
    ///
    /// # Arguments
    /// * `text` - Input text
    /// * `options` - Optional configuration options
    ///
    /// # Returns
    /// Vector of sentences
    ///
    /// # Example
    /// ```no_run
    /// use proteus::wtpsplit::SaT;
    ///
    /// let mut sat = SaT::new("sat-3l-sm", None)?;
    /// let sentences = sat.split("Hello world. This is a test.", None)?;
    /// assert_eq!(sentences.len(), 2);
    /// # Ok::<(), proteus::wtpsplit::Error>(())
    /// ```
    pub fn split(&mut self, text: &str, options: Option<&SaTOptions>) -> Result<Vec<String>> {
        let options = options.cloned().unwrap_or_default();
        let results = self.split_batch(&[text], Some(&options))?;
        Ok(results.into_iter().next().unwrap_or_default())
    }

    /// Split a batch of texts into sentences
    pub fn split_batch(
        &mut self,
        texts: &[&str],
        options: Option<&SaTOptions>,
    ) -> Result<Vec<Vec<String>>> {
        let options = options.cloned().unwrap_or_default();
        let threshold = options.threshold.unwrap_or_else(|| self.get_default_threshold());

        let all_probs = self.predict_proba_batch(texts, Some(&options))?;

        let results: Vec<Vec<String>> = texts
            .iter()
            .zip(all_probs.iter())
            .map(|(text, probs)| {
                // Find indices where probability exceeds threshold
                let indices: Vec<usize> = probs
                    .iter()
                    .enumerate()
                    .filter(|(_, &p)| p > threshold)
                    .map(|(i, _)| i)
                    .collect();

                // Convert to sentences
                let mut sentences = indices_to_sentences(text, &indices, options.strip_whitespace);

                // Split on input newlines if requested
                if options.split_on_input_newlines {
                    sentences = sentences
                        .into_iter()
                        .flat_map(|s| s.split('\n').map(String::from).collect::<Vec<_>>())
                        .filter(|s| !s.is_empty())
                        .collect();
                }

                sentences
            })
            .collect();

        Ok(results)
    }

    /// Split text into paragraphs, each containing sentences
    ///
    /// # Returns
    /// Vector of paragraphs, where each paragraph is a vector of sentences
    pub fn split_paragraphs(
        &mut self,
        text: &str,
        options: Option<&SaTOptions>,
    ) -> Result<Vec<Vec<String>>> {
        let mut options = options.cloned().unwrap_or_default();
        options.do_paragraph_segmentation = true;

        let probs = self.predict_proba(text, Some(&options))?;
        let threshold = options.threshold.unwrap_or_else(|| self.get_default_threshold());

        // First split into paragraphs using paragraph threshold
        let para_indices: Vec<usize> = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > options.paragraph_threshold)
            .map(|(i, _)| i)
            .collect();

        let paragraphs = indices_to_sentences(text, &para_indices, false);

        // Then split each paragraph into sentences
        let mut offset = 0;
        let mut result = Vec::new();

        for paragraph in paragraphs {
            let para_len = paragraph.chars().count();
            let para_probs: Vec<f32> = probs[offset..offset + para_len].to_vec();

            let sent_indices: Vec<usize> = para_probs
                .iter()
                .enumerate()
                .filter(|(_, &p)| p > threshold)
                .map(|(i, _)| i)
                .collect();

            let sentences = indices_to_sentences(&paragraph, &sent_indices, options.strip_whitespace);
            result.push(sentences);

            offset += para_len;
        }

        Ok(result)
    }

    /// Get the model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.model.config
    }
}
