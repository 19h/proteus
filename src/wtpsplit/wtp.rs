//! WtP (Where's the Point) - Legacy character-based sentence segmentation
//!
//! WtP models use character-level hash encoding and are considered deprecated.
//! For new applications, use SaT models instead.

use std::path::Path;

use crate::wtpsplit::config::ModelConfig;
use crate::wtpsplit::extract::{extract_wtp, logits_to_probs, ExtractConfig, Weighting};
use crate::wtpsplit::hub::{download_model, get_local_model_files, is_local_path, WTP_HUB_PREFIX};
use crate::wtpsplit::model::OnnxModel;
use crate::wtpsplit::utils::{indices_to_sentences, reinsert_space_probs, remove_spaces};
use crate::wtpsplit::Result;

/// Configuration options for WtP splitting
#[derive(Debug, Clone)]
pub struct WtPOptions {
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
    /// Remove whitespace before inference
    pub remove_whitespace_before_inference: bool,
    /// Paragraph threshold
    pub paragraph_threshold: f32,
    /// Strip whitespace from sentences
    pub strip_whitespace: bool,
    /// Perform paragraph segmentation
    pub do_paragraph_segmentation: bool,
    /// Language code (for language adapters, if available)
    pub lang_code: Option<String>,
    /// Style (for mixture models, if available)
    pub style: Option<String>,
}

impl Default for WtPOptions {
    fn default() -> Self {
        Self {
            threshold: None,
            stride: 256,
            block_size: 512,
            batch_size: 32,
            pad_last_batch: false,
            weighting: Weighting::Uniform,
            remove_whitespace_before_inference: false,
            paragraph_threshold: 0.5,
            strip_whitespace: false,
            do_paragraph_segmentation: false,
            lang_code: None,
            style: None,
        }
    }
}

/// WtP sentence segmentation model (deprecated, use SaT instead)
#[deprecated(
    since = "0.1.0",
    note = "WtP models are deprecated. Please use SaT models for better performance."
)]
#[allow(dead_code)]
pub struct WtP {
    model: OnnxModel,
    model_name: String,
}

#[allow(deprecated)]
impl WtP {
    /// Create a new WtP instance from a model name or path
    ///
    /// # Arguments
    /// * `model_name_or_path` - Either a model name (e.g., "wtp-bert-mini") or a local path
    /// * `hub_prefix` - Optional hub prefix (defaults to "benjamin")
    ///
    /// # Note
    /// WtP models are deprecated. Consider using SaT models instead.
    pub fn new(model_name_or_path: &str, hub_prefix: Option<&str>) -> Result<Self> {
        log::warn!(
            "WtP models are deprecated. Consider using SaT models for better performance and efficiency."
        );

        let hub_prefix = hub_prefix.unwrap_or(WTP_HUB_PREFIX);

        // Get model files
        let model_files = if is_local_path(model_name_or_path) {
            get_local_model_files(Path::new(model_name_or_path), false)?
        } else {
            download_model(model_name_or_path, Some(hub_prefix), false)?
        };

        // Load config
        let config = ModelConfig::from_file(&model_files.config_path)?;

        // Load ONNX model
        let model = OnnxModel::new(&model_files.onnx_path, config)?;

        Ok(Self {
            model,
            model_name: model_name_or_path.to_string(),
        })
    }

    /// Get the default threshold for WtP models
    fn get_default_threshold(&self) -> f32 {
        0.01
    }

    /// Get sentence boundary probabilities for a text
    pub fn predict_proba(&mut self, text: &str, options: Option<&WtPOptions>) -> Result<Vec<f32>> {
        let options = options.cloned().unwrap_or_default();
        let results = self.predict_proba_batch(&[text], Some(&options))?;
        Ok(results.into_iter().next().unwrap_or_default())
    }

    /// Get sentence boundary probabilities for a batch of texts
    pub fn predict_proba_batch(
        &mut self,
        texts: &[&str],
        options: Option<&WtPOptions>,
    ) -> Result<Vec<Vec<f32>>> {
        let options = options.cloned().unwrap_or_default();

        // Handle whitespace removal
        let (input_texts, space_positions): (Vec<String>, Vec<Vec<usize>>) =
            if options.remove_whitespace_before_inference {
                texts.iter().map(|t| remove_spaces(t)).unzip()
            } else {
                (
                    texts.iter().map(|s| s.to_string()).collect(),
                    vec![vec![]; texts.len()],
                )
            };

        // Filter empty strings
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
            extract_wtp(&non_empty_texts, &mut self.model, &extract_config)?
        };

        // Convert to probabilities
        let mut all_probs: Vec<Vec<f32>> = vec![vec![]; texts.len()];

        for (result_idx, &text_idx) in non_empty_indices.iter().enumerate() {
            let probs = logits_to_probs(&extraction_result.logits[result_idx]);
            all_probs[text_idx] = probs;
        }

        // Fill empty texts
        for i in 0..texts.len() {
            if all_probs[i].is_empty() && !texts[i].is_empty() {
                all_probs[i] = vec![0.0; texts[i].chars().count()];
            }
        }

        // Reinsert spaces if needed
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
    pub fn split(&mut self, text: &str, options: Option<&WtPOptions>) -> Result<Vec<String>> {
        let options = options.cloned().unwrap_or_default();
        let results = self.split_batch(&[text], Some(&options))?;
        Ok(results.into_iter().next().unwrap_or_default())
    }

    /// Split a batch of texts into sentences
    pub fn split_batch(
        &mut self,
        texts: &[&str],
        options: Option<&WtPOptions>,
    ) -> Result<Vec<Vec<String>>> {
        let options = options.cloned().unwrap_or_default();
        let threshold = options.threshold.unwrap_or_else(|| self.get_default_threshold());

        let all_probs = self.predict_proba_batch(texts, Some(&options))?;

        let results: Vec<Vec<String>> = texts
            .iter()
            .zip(all_probs.iter())
            .map(|(text, probs)| {
                let indices: Vec<usize> = probs
                    .iter()
                    .enumerate()
                    .filter(|(_, &p)| p > threshold)
                    .map(|(i, _)| i)
                    .collect();

                indices_to_sentences(text, &indices, options.strip_whitespace)
            })
            .collect();

        Ok(results)
    }

    /// Get the model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.model.config
    }
}
