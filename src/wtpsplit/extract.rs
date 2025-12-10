//! Core extraction logic for sentence segmentation
//!
//! This module handles:
//! - Text chunking with overlapping windows
//! - Batch processing through the model
//! - Logit aggregation across overlapping chunks

use ndarray::{s, Array2, Array3};
use tokenizers::Tokenizer;
use crate::wtpsplit::constants::NEWLINE_INDEX;
use crate::wtpsplit::model::OnnxModel;
use crate::wtpsplit::utils::{hash_encode, sigmoid, token_to_char_probs};
use crate::wtpsplit::Result;

/// Weighting scheme for aggregating overlapping chunk predictions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weighting {
    /// All positions weighted equally
    Uniform,
    /// Triangular (hat) weighting - higher weight in the center
    Hat,
}

impl Default for Weighting {
    fn default() -> Self {
        Weighting::Uniform
    }
}

/// Configuration for extraction
#[derive(Debug, Clone)]
pub struct ExtractConfig {
    /// Maximum block size for chunking
    pub block_size: usize,
    /// Stride for overlapping chunks
    pub stride: usize,
    /// Batch size for inference
    pub batch_size: usize,
    /// Whether to pad the last batch
    pub pad_last_batch: bool,
    /// Weighting scheme for overlapping predictions
    pub weighting: Weighting,
}

impl Default for ExtractConfig {
    fn default() -> Self {
        Self {
            block_size: 512,
            stride: 64,
            batch_size: 32,
            pad_last_batch: false,
            weighting: Weighting::Uniform,
        }
    }
}

/// Chunk location tracking
#[derive(Debug, Clone, Copy)]
struct ChunkLoc {
    text_idx: usize,
    start: usize,
    end: usize,
}

/// Result of extraction containing logits for each text
pub struct ExtractionResult {
    /// Logits for each text (character-level)
    pub logits: Vec<Vec<Vec<f32>>>,
    /// Offset mappings for subword models (token -> char spans)
    pub offset_mappings: Option<Vec<Vec<(usize, usize)>>>,
}

/// Extract logits from a batch of texts using a SaT model
///
/// This function:
/// 1. Tokenizes and chunks the texts
/// 2. Runs inference on each chunk
/// 3. Aggregates overlapping predictions
/// 4. Maps token-level predictions back to characters
pub fn extract_sat(
    texts: &[&str],
    model: &mut OnnxModel,
    tokenizer: &Tokenizer,
    config: &ExtractConfig,
) -> Result<ExtractionResult> {
    if texts.is_empty() {
        return Ok(ExtractionResult {
            logits: vec![],
            offset_mappings: None,
        });
    }

    // Tokenize all texts
    let encodings: Vec<_> = texts
        .iter()
        .map(|text| {
            tokenizer
                .encode(*text, false)
                .expect("Tokenization failed")
        })
        .collect();

    // Get token IDs and offset mappings
    let token_ids: Vec<Vec<u32>> = encodings.iter().map(|e| e.get_ids().to_vec()).collect();
    let offset_mappings: Vec<Vec<(usize, usize)>> = encodings
        .iter()
        .map(|e| {
            e.get_offsets()
                .iter()
                .map(|&(start, end)| (start, end))
                .collect()
        })
        .collect();

    let text_lengths: Vec<usize> = token_ids.iter().map(|t| t.len()).collect();

    // Determine effective block size (accounting for CLS/SEP tokens)
    let max_len = text_lengths.iter().max().copied().unwrap_or(0);
    let mut block_size = config.block_size.min(max_len);

    // Account for CLS and SEP tokens (2 extra tokens)
    if block_size > 510 {
        block_size = 510;
    }

    // Adjust for downsampling rate
    let downsampling_rate = model.config.downsampling_rate;
    block_size = ((block_size + downsampling_rate - 1) / downsampling_rate) * downsampling_rate;

    // Calculate total number of chunks
    let num_chunks: usize = text_lengths
        .iter()
        .map(|&len| {
            if len <= block_size {
                1
            } else {
                ((len - block_size + config.stride - 1) / config.stride) + 1
            }
        })
        .sum();

    // CLS and SEP token IDs for XLM-RoBERTa
    let cls_token_id = 0u32; // <s>
    let sep_token_id = 2u32; // </s>
    let _pad_token_id = 1u32; // <pad>

    // Preallocate input arrays (with space for CLS and SEP)
    let seq_len = block_size + 2;
    let mut input_ids = Array2::<i64>::zeros((num_chunks, seq_len));
    let mut attention_mask = Array2::<f32>::zeros((num_chunks, seq_len));
    let mut locs = Vec::with_capacity(num_chunks);

    // Fill input arrays
    let mut current_chunk = 0;
    for (text_idx, tokens) in token_ids.iter().enumerate() {
        let text_len = tokens.len();
        let mut j = 0;

        loop {
            let mut start = j;
            let mut end = j + block_size;
            let done = end >= text_len;

            // If this chunk extends beyond text, adjust to cover the end
            if done {
                end = text_len;
                start = end.saturating_sub(block_size);
            }

            // Build chunk with CLS and SEP
            let chunk_tokens = &tokens[start..end];
            let chunk_len = chunk_tokens.len();

            input_ids[[current_chunk, 0]] = cls_token_id as i64;
            for (i, &tok) in chunk_tokens.iter().enumerate() {
                input_ids[[current_chunk, i + 1]] = tok as i64;
            }
            input_ids[[current_chunk, chunk_len + 1]] = sep_token_id as i64;

            // Set attention mask
            for i in 0..(chunk_len + 2) {
                attention_mask[[current_chunk, i]] = 1.0;
            }

            locs.push(ChunkLoc {
                text_idx,
                start,
                end,
            });
            current_chunk += 1;

            if done {
                break;
            }

            j += config.stride;
        }
    }

    // Truncate to actual number of chunks created
    let num_chunks = current_chunk;
    let input_ids = input_ids.slice(s![..num_chunks, ..]).to_owned();
    let attention_mask = attention_mask.slice(s![..num_chunks, ..]).to_owned();
    let locs = &locs[..num_chunks];

    // Compute weights for aggregation
    let weights: Vec<f32> = match config.weighting {
        Weighting::Uniform => vec![1.0; block_size],
        Weighting::Hat => {
            let mut w = Vec::with_capacity(block_size);
            for i in 0..block_size {
                let x = (2.0 * i as f32 / (block_size - 1) as f32) - 1.0;
                w.push(1.0 - x.abs());
            }
            w
        }
    };

    // Prepare output containers
    let num_labels = model.config.num_labels;
    let mut all_logits: Vec<Vec<Vec<f32>>> = text_lengths
        .iter()
        .map(|&len| vec![vec![0.0; num_labels]; len])
        .collect();
    let mut all_counts: Vec<Vec<f32>> = text_lengths.iter().map(|&len| vec![0.0; len]).collect();

    // Process in batches
    let n_batches = (num_chunks + config.batch_size - 1) / config.batch_size;

    for batch_idx in 0..n_batches {
        let start = batch_idx * config.batch_size;
        let end = (start + config.batch_size).min(num_chunks);

        let batch_input_ids = input_ids.slice(s![start..end, ..]);
        let batch_attention_mask = attention_mask.slice(s![start..end, ..]);

        // Run inference
        let batch_logits = model.forward_sat(batch_input_ids, batch_attention_mask)?;

        // Remove CLS and SEP token predictions (first and last)
        let batch_logits = batch_logits.slice(s![.., 1..-1, ..]);

        // Aggregate predictions
        for (i, chunk_idx) in (start..end).enumerate() {
            let loc = &locs[chunk_idx];
            let n = loc.end - loc.start;

            for j in 0..n {
                let char_idx = loc.start + j;
                let weight = weights[j];

                for label in 0..num_labels {
                    all_logits[loc.text_idx][char_idx][label] +=
                        weight * batch_logits[[i, j, label]];
                }
                all_counts[loc.text_idx][char_idx] += weight;
            }
        }
    }

    // Average the logits
    for (text_logits, text_counts) in all_logits.iter_mut().zip(all_counts.iter()) {
        for (char_logits, &count) in text_logits.iter_mut().zip(text_counts.iter()) {
            if count > 0.0 {
                for logit in char_logits.iter_mut() {
                    *logit /= count;
                }
            }
        }
    }

    // Convert token-level logits to character-level
    let char_level_logits: Vec<Vec<Vec<f32>>> = texts
        .iter()
        .zip(all_logits.iter())
        .zip(offset_mappings.iter())
        .map(|((text, token_logits), offsets)| {
            token_to_char_probs(text.chars().count(), token_logits, offsets, num_labels)
        })
        .collect();

    Ok(ExtractionResult {
        logits: char_level_logits,
        offset_mappings: Some(offset_mappings),
    })
}

/// Extract logits from a batch of texts using a WtP model (character-based)
pub fn extract_wtp(
    texts: &[&str],
    model: &mut OnnxModel,
    config: &ExtractConfig,
) -> Result<ExtractionResult> {
    if texts.is_empty() {
        return Ok(ExtractionResult {
            logits: vec![],
            offset_mappings: None,
        });
    }

    // Encode texts as character ordinals
    let encoded_texts: Vec<Vec<i64>> = texts
        .iter()
        .map(|text| text.chars().map(|c| c as i64).collect())
        .collect();

    let text_lengths: Vec<usize> = encoded_texts.iter().map(|t| t.len()).collect();

    // Determine effective block size
    let max_len = text_lengths.iter().max().copied().unwrap_or(0);
    let mut block_size = config.block_size.min(max_len);

    // Adjust for downsampling rate
    let downsampling_rate = model.config.downsampling_rate;
    block_size = ((block_size + downsampling_rate - 1) / downsampling_rate) * downsampling_rate;

    // Calculate total number of chunks
    let num_chunks: usize = text_lengths
        .iter()
        .map(|&len| {
            if len <= block_size {
                1
            } else {
                ((len - block_size + config.stride - 1) / config.stride) + 1
            }
        })
        .sum();

    // Hash encode all characters
    let num_hashes = model.config.num_hash_functions;
    let num_buckets = model.config.num_hash_buckets as i64;

    let all_hashes: Vec<Vec<Vec<i64>>> = encoded_texts
        .iter()
        .map(|ordinals| hash_encode(ordinals, num_hashes, num_buckets))
        .collect();

    // Preallocate input arrays
    let mut hashed_ids = Array3::<i64>::zeros((num_chunks, block_size, num_hashes));
    let mut attention_mask = Array2::<f32>::zeros((num_chunks, block_size));
    let mut locs = Vec::with_capacity(num_chunks);

    // Fill input arrays
    let mut current_chunk = 0;
    for (text_idx, hashes) in all_hashes.iter().enumerate() {
        let text_len = hashes.len();
        let mut j = 0;

        loop {
            let start = j;
            let mut end = (j + block_size).min(text_len);
            let done = end >= text_len;

            if done {
                let new_start = end.saturating_sub(block_size);
                let start = new_start;
                end = text_len;

                for (i, hash_row) in hashes[start..end].iter().enumerate() {
                    for (h, &hash_val) in hash_row.iter().enumerate() {
                        hashed_ids[[current_chunk, i, h]] = hash_val;
                    }
                    attention_mask[[current_chunk, i]] = 1.0;
                }

                locs.push(ChunkLoc {
                    text_idx,
                    start,
                    end,
                });
                current_chunk += 1;
                break;
            }

            for (i, hash_row) in hashes[start..end].iter().enumerate() {
                for (h, &hash_val) in hash_row.iter().enumerate() {
                    hashed_ids[[current_chunk, i, h]] = hash_val;
                }
                attention_mask[[current_chunk, i]] = 1.0;
            }

            locs.push(ChunkLoc {
                text_idx,
                start,
                end,
            });
            current_chunk += 1;

            j += config.stride;
            if j >= text_len {
                break;
            }
        }
    }

    // Truncate to actual number of chunks
    let num_chunks = current_chunk;
    let hashed_ids = hashed_ids.slice(s![..num_chunks, .., ..]).to_owned();
    let attention_mask = attention_mask.slice(s![..num_chunks, ..]).to_owned();
    let locs = &locs[..num_chunks];

    // Compute weights
    let weights: Vec<f32> = match config.weighting {
        Weighting::Uniform => vec![1.0; block_size],
        Weighting::Hat => {
            let mut w = Vec::with_capacity(block_size);
            for i in 0..block_size {
                let x = (2.0 * i as f32 / (block_size - 1) as f32) - 1.0;
                w.push(1.0 - x.abs());
            }
            w
        }
    };

    // Prepare output containers
    let num_labels = model.config.num_labels;
    let mut all_logits: Vec<Vec<Vec<f32>>> = text_lengths
        .iter()
        .map(|&len| vec![vec![0.0; num_labels]; len])
        .collect();
    let mut all_counts: Vec<Vec<f32>> = text_lengths.iter().map(|&len| vec![0.0; len]).collect();

    // Process in batches
    let n_batches = (num_chunks + config.batch_size - 1) / config.batch_size;

    for batch_idx in 0..n_batches {
        let start = batch_idx * config.batch_size;
        let end = (start + config.batch_size).min(num_chunks);

        let batch_hashed_ids = hashed_ids.slice(s![start..end, .., ..]);
        let batch_attention_mask = attention_mask.slice(s![start..end, ..]);

        // Run inference
        let batch_logits = model.forward_wtp(batch_hashed_ids, batch_attention_mask)?;

        // Aggregate predictions
        for (i, chunk_idx) in (start..end).enumerate() {
            let loc = &locs[chunk_idx];
            let n = loc.end - loc.start;

            for j in 0..n {
                let char_idx = loc.start + j;
                let weight = weights[j];

                for label in 0..num_labels {
                    all_logits[loc.text_idx][char_idx][label] +=
                        weight * batch_logits[[i, j, label]];
                }
                all_counts[loc.text_idx][char_idx] += weight;
            }
        }
    }

    // Average the logits
    for (text_logits, text_counts) in all_logits.iter_mut().zip(all_counts.iter()) {
        for (char_logits, &count) in text_logits.iter_mut().zip(text_counts.iter()) {
            if count > 0.0 {
                for logit in char_logits.iter_mut() {
                    *logit /= count;
                }
            }
        }
    }

    Ok(ExtractionResult {
        logits: all_logits,
        offset_mappings: None,
    })
}

/// Convert logits to sentence boundary probabilities
pub fn logits_to_probs(logits: &[Vec<f32>]) -> Vec<f32> {
    logits
        .iter()
        .map(|char_logits| {
            if char_logits.len() > NEWLINE_INDEX {
                sigmoid(char_logits[NEWLINE_INDEX])
            } else {
                0.0
            }
        })
        .collect()
}
