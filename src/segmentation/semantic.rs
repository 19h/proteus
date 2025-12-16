//! Semantic text segmentation using TextTiling with SDR fingerprints.
//!
//! This implements the TextTiling algorithm (Hearst, 1997) adapted for
//! Sparse Distributed Representations. The algorithm detects topic boundaries
//! by analyzing similarity patterns between adjacent text blocks.

use crate::error::Result;
use crate::fingerprint::Sdr;
use crate::storage::Retina;
use crate::text::SentenceSegmenter;
use std::time::Instant;

/// Configuration for semantic segmentation.
#[derive(Debug, Clone)]
pub struct SegmentationConfig {
    /// Number of sentences per block for comparison (default: 3).
    pub block_size: usize,
    /// Depth threshold in standard deviations above mean (default: 0.5).
    pub depth_threshold: f32,
    /// Minimum sentences per segment (default: 2).
    pub min_segment_sentences: usize,
    /// Apply Gaussian smoothing to depth scores (default: true).
    pub smoothing: bool,
    /// Sigma for Gaussian smoothing (default: 1.0).
    pub smoothing_sigma: f32,
    /// Use fast regex-based sentence splitting instead of neural model (default: false).
    pub fast_mode: bool,
    /// Print debug/progress information during segmentation (default: false).
    pub verbose: bool,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            block_size: 3,
            depth_threshold: 0.5,
            min_segment_sentences: 2,
            smoothing: true,
            smoothing_sigma: 1.0,
            fast_mode: false,
            verbose: false,
        }
    }
}

/// A semantic segment of text.
#[derive(Debug, Clone)]
pub struct SemanticSegment {
    /// Individual sentences in this segment.
    pub sentences: Vec<String>,
    /// Joined text of all sentences.
    pub text: String,
    /// Index of first sentence (0-based).
    pub start_idx: usize,
    /// Index after last sentence (exclusive).
    pub end_idx: usize,
}

/// Result of semantic segmentation with metadata.
#[derive(Debug, Clone)]
pub struct SegmentationResult {
    /// The semantic segments.
    pub segments: Vec<SemanticSegment>,
    /// Sentence indices where boundaries occur.
    pub boundary_indices: Vec<usize>,
    /// Raw similarity scores between adjacent blocks.
    pub similarity_scores: Vec<f32>,
    /// Computed depth scores at each gap.
    pub depth_scores: Vec<f32>,
    /// Total number of sentences.
    pub num_sentences: usize,
    /// Number of blocks created.
    pub num_blocks: usize,
    /// Mean of depth scores.
    pub depth_mean: f32,
    /// Standard deviation of depth scores.
    pub depth_std: f32,
}

/// Semantic text segmenter using TextTiling algorithm.
pub struct SemanticSegmenter {
    retina: Retina,
    sentence_segmenter: Option<SentenceSegmenter>,
    config: SegmentationConfig,
}

impl SemanticSegmenter {
    /// Creates a new semantic segmenter with default configuration.
    pub fn new(retina: Retina) -> Result<Self> {
        Self::with_config(retina, SegmentationConfig::default())
    }

    /// Creates a new semantic segmenter with custom configuration.
    pub fn with_config(retina: Retina, config: SegmentationConfig) -> Result<Self> {
        // Only load the neural model if not in fast mode
        let sentence_segmenter = if config.fast_mode {
            if config.verbose {
                eprintln!("[segmenter] Using fast regex-based sentence splitting");
            }
            None
        } else {
            if config.verbose {
                eprintln!("[segmenter] Loading neural sentence model (sat-3l-sm)...");
                let start = Instant::now();
                let seg = SentenceSegmenter::new("sat-3l-sm")?;
                eprintln!("[segmenter] Model loaded in {:?}", start.elapsed());
                Some(seg)
            } else {
                Some(SentenceSegmenter::new("sat-3l-sm")?)
            }
        };
        Ok(Self {
            retina,
            sentence_segmenter,
            config,
        })
    }

    /// Fast regex-based sentence splitting.
    /// Splits on .!? followed by whitespace or end of string.
    fn fast_sentence_split(text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();
        let mut chars = text.chars().peekable();

        while let Some(c) = chars.next() {
            current.push(c);

            // Check for sentence-ending punctuation
            if c == '.' || c == '!' || c == '?' {
                // Look ahead - is it followed by whitespace, newline, or end?
                match chars.peek() {
                    None => {
                        // End of text
                        let trimmed = current.trim().to_string();
                        if !trimmed.is_empty() {
                            sentences.push(trimmed);
                        }
                        current.clear();
                    }
                    Some(&next) if next.is_whitespace() => {
                        // Sentence boundary
                        let trimmed = current.trim().to_string();
                        if !trimmed.is_empty() {
                            sentences.push(trimmed);
                        }
                        current.clear();
                    }
                    _ => {
                        // Not a sentence boundary (e.g., "Dr." or "3.14")
                    }
                }
            }
        }

        // Don't forget any remaining text
        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() {
            sentences.push(trimmed);
        }

        sentences
    }

    /// Performs semantic segmentation on the given text.
    ///
    /// Returns a full result including segments and analysis metadata.
    pub fn segment(&mut self, text: &str) -> Result<SegmentationResult> {
        let total_start = Instant::now();

        // Step 1: Sentence segmentation
        if self.config.verbose {
            eprintln!("[segment] Step 1: Sentence segmentation...");
        }
        let step_start = Instant::now();

        let sentences = if let Some(ref mut segmenter) = self.sentence_segmenter {
            segmenter.segment(text)?
        } else {
            Self::fast_sentence_split(text)
        };

        if self.config.verbose {
            eprintln!("[segment]   Found {} sentences in {:?}", sentences.len(), step_start.elapsed());
        }

        if sentences.is_empty() {
            return Ok(SegmentationResult {
                segments: vec![],
                boundary_indices: vec![],
                similarity_scores: vec![],
                depth_scores: vec![],
                num_sentences: 0,
                num_blocks: 0,
                depth_mean: 0.0,
                depth_std: 0.0,
            });
        }

        // Handle case where text is too short for blocking
        if sentences.len() < self.config.block_size * 2 {
            if self.config.verbose {
                eprintln!("[segment] Text too short for blocking ({} sentences < {} min), returning single segment",
                    sentences.len(), self.config.block_size * 2);
            }
            let segment = SemanticSegment {
                text: sentences.join(" "),
                sentences: sentences.clone(),
                start_idx: 0,
                end_idx: sentences.len(),
            };
            return Ok(SegmentationResult {
                segments: vec![segment],
                boundary_indices: vec![],
                similarity_scores: vec![],
                depth_scores: vec![],
                num_sentences: sentences.len(),
                num_blocks: 1,
                depth_mean: 0.0,
                depth_std: 0.0,
            });
        }

        // Step 2: Compute fingerprints for each sentence
        if self.config.verbose {
            eprintln!("[segment] Step 2: Computing fingerprints for {} sentences...", sentences.len());
        }
        let step_start = Instant::now();

        let sentence_fingerprints: Vec<Sdr> = sentences
            .iter()
            .map(|s| self.retina.fingerprint_text(s))
            .collect();

        if self.config.verbose {
            let avg_bits: f64 = sentence_fingerprints.iter()
                .map(|fp| fp.cardinality() as f64)
                .sum::<f64>() / sentence_fingerprints.len() as f64;
            eprintln!("[segment]   Fingerprinted in {:?} (avg {:.1} active bits/sentence)",
                step_start.elapsed(), avg_bits);
        }

        // Step 3: Create blocks and compute block fingerprints
        if self.config.verbose {
            eprintln!("[segment] Step 3: Creating {} blocks ({} sentences/block)...",
                sentences.len() / self.config.block_size, self.config.block_size);
        }
        let step_start = Instant::now();

        let block_size = self.config.block_size;
        let num_blocks = sentences.len() / block_size;

        let block_fingerprints: Vec<Sdr> = (0..num_blocks)
            .map(|i| {
                let start = i * block_size;
                let end = (start + block_size).min(sentences.len());
                self.combine_fingerprints(&sentence_fingerprints[start..end])
            })
            .collect();

        if self.config.verbose {
            eprintln!("[segment]   Created {} block fingerprints in {:?}",
                block_fingerprints.len(), step_start.elapsed());
        }

        // Step 4: Compute similarity scores between adjacent blocks
        if self.config.verbose {
            eprintln!("[segment] Step 4: Computing {} similarity scores...", block_fingerprints.len().saturating_sub(1));
        }
        let step_start = Instant::now();

        let similarity_scores: Vec<f32> = (0..block_fingerprints.len().saturating_sub(1))
            .map(|i| {
                block_fingerprints[i].cosine_similarity(&block_fingerprints[i + 1]) as f32
            })
            .collect();

        if self.config.verbose {
            if !similarity_scores.is_empty() {
                let min = similarity_scores.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = similarity_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let avg = similarity_scores.iter().sum::<f32>() / similarity_scores.len() as f32;
                eprintln!("[segment]   Similarities in {:?} (min={:.3}, avg={:.3}, max={:.3})",
                    step_start.elapsed(), min, avg, max);
            }
        }

        if similarity_scores.is_empty() {
            let segment = SemanticSegment {
                text: sentences.join(" "),
                sentences: sentences.clone(),
                start_idx: 0,
                end_idx: sentences.len(),
            };
            return Ok(SegmentationResult {
                segments: vec![segment],
                boundary_indices: vec![],
                similarity_scores: vec![],
                depth_scores: vec![],
                num_sentences: sentences.len(),
                num_blocks,
                depth_mean: 0.0,
                depth_std: 0.0,
            });
        }

        // Step 5: Compute depth scores
        if self.config.verbose {
            eprintln!("[segment] Step 5: Computing depth scores...");
        }
        let step_start = Instant::now();

        let mut depth_scores = self.compute_depth_scores(&similarity_scores);

        if self.config.verbose {
            eprintln!("[segment]   Depth scores computed in {:?}", step_start.elapsed());
        }

        // Step 6: Apply Gaussian smoothing if enabled
        if self.config.smoothing && depth_scores.len() > 2 {
            if self.config.verbose {
                eprintln!("[segment] Step 6: Applying Gaussian smoothing (sigma={})...", self.config.smoothing_sigma);
            }
            let step_start = Instant::now();
            depth_scores = self.gaussian_smooth(&depth_scores, self.config.smoothing_sigma);
            if self.config.verbose {
                eprintln!("[segment]   Smoothing applied in {:?}", step_start.elapsed());
            }
        } else if self.config.verbose {
            eprintln!("[segment] Step 6: Skipping smoothing (disabled or too few scores)");
        }

        // Step 7: Compute threshold for boundary detection
        if self.config.verbose {
            eprintln!("[segment] Step 7: Computing boundary threshold...");
        }

        let (depth_mean, depth_std) = self.mean_std(&depth_scores);
        let threshold = depth_mean + self.config.depth_threshold * depth_std;

        if self.config.verbose {
            eprintln!("[segment]   Depth mean={:.4}, std={:.4}, threshold={:.4} ({:.1} std above mean)",
                depth_mean, depth_std, threshold, self.config.depth_threshold);
        }

        // Step 8: Find boundary indices (in terms of blocks, then convert to sentences)
        if self.config.verbose {
            eprintln!("[segment] Step 8: Finding boundaries...");
        }

        let mut block_boundaries: Vec<usize> = depth_scores
            .iter()
            .enumerate()
            .filter(|(_, &d)| d > threshold)
            .map(|(i, _)| i + 1) // +1 because boundary is after block i
            .collect();

        if self.config.verbose {
            eprintln!("[segment]   Found {} raw boundaries above threshold", block_boundaries.len());
        }

        // Enforce minimum segment size
        block_boundaries = self.enforce_min_segment_size(
            &block_boundaries,
            num_blocks,
            self.config.min_segment_sentences / block_size.max(1),
        );

        if self.config.verbose {
            eprintln!("[segment]   After enforcing min segment size: {} boundaries", block_boundaries.len());
        }

        // Convert block boundaries to sentence boundaries
        let boundary_indices: Vec<usize> = block_boundaries
            .iter()
            .map(|&b| (b * block_size).min(sentences.len()))
            .collect();

        // Step 9: Create segments
        if self.config.verbose {
            eprintln!("[segment] Step 9: Creating {} segments...", boundary_indices.len() + 1);
        }

        let segments = self.create_segments(&sentences, &boundary_indices);

        if self.config.verbose {
            eprintln!("[segment] Complete! Total time: {:?}", total_start.elapsed());
            eprintln!("[segment] Summary: {} sentences -> {} blocks -> {} segments",
                sentences.len(), num_blocks, segments.len());
        }

        Ok(SegmentationResult {
            segments,
            boundary_indices,
            similarity_scores,
            depth_scores,
            num_sentences: sentences.len(),
            num_blocks,
            depth_mean,
            depth_std,
        })
    }

    /// Convenience method that returns just the segments.
    pub fn segment_text(&mut self, text: &str) -> Result<Vec<SemanticSegment>> {
        Ok(self.segment(text)?.segments)
    }

    /// Combines multiple fingerprints into one using union.
    fn combine_fingerprints(&self, fingerprints: &[Sdr]) -> Sdr {
        if fingerprints.is_empty() {
            return Sdr::new(self.retina.grid_size());
        }

        let mut combined = fingerprints[0].clone();
        for fp in &fingerprints[1..] {
            combined = combined.union(fp);
        }

        // Sparsify to maintain reasonable density
        let max_bits = self.retina.config().max_active_bits;
        combined.sparsify(max_bits);
        combined
    }

    /// Computes depth scores from similarity scores.
    ///
    /// Depth at position i measures how much the similarity dips compared
    /// to the peaks on either side.
    fn compute_depth_scores(&self, similarities: &[f32]) -> Vec<f32> {
        let n = similarities.len();
        if n == 0 {
            return vec![];
        }

        let mut depths = vec![0.0f32; n];

        for i in 0..n {
            // Find left peak (maximum similarity to the left, including current)
            let left_peak = similarities[..=i]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);

            // Find right peak (maximum similarity to the right, including current)
            let right_peak = similarities[i..]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);

            // Depth is how much current point dips below surrounding peaks
            depths[i] = (left_peak - similarities[i]) + (right_peak - similarities[i]);
        }

        depths
    }

    /// Applies 1D Gaussian smoothing to the scores.
    fn gaussian_smooth(&self, scores: &[f32], sigma: f32) -> Vec<f32> {
        if scores.len() < 3 {
            return scores.to_vec();
        }

        // Compute kernel size (3 sigma rule)
        let kernel_radius = (3.0 * sigma).ceil() as usize;
        let kernel_size = 2 * kernel_radius + 1;

        // Generate Gaussian kernel
        let mut kernel = vec![0.0f32; kernel_size];
        let sigma_sq = sigma * sigma;
        let mut sum = 0.0f32;

        for i in 0..kernel_size {
            let x = i as f32 - kernel_radius as f32;
            kernel[i] = (-x * x / (2.0 * sigma_sq)).exp();
            sum += kernel[i];
        }

        // Normalize kernel
        for k in &mut kernel {
            *k /= sum;
        }

        // Apply convolution with edge padding
        let mut smoothed = vec![0.0f32; scores.len()];

        for i in 0..scores.len() {
            let mut value = 0.0f32;
            for (j, &k) in kernel.iter().enumerate() {
                let idx = i as isize + j as isize - kernel_radius as isize;
                let clamped_idx = idx.clamp(0, scores.len() as isize - 1) as usize;
                value += k * scores[clamped_idx];
            }
            smoothed[i] = value;
        }

        smoothed
    }

    /// Computes mean and standard deviation of a slice.
    fn mean_std(&self, values: &[f32]) -> (f32, f32) {
        if values.is_empty() {
            return (0.0, 0.0);
        }

        let n = values.len() as f32;
        let mean = values.iter().sum::<f32>() / n;

        let variance = values.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
        let std = variance.sqrt();

        (mean, std)
    }

    /// Enforces minimum segment size by removing closely-spaced boundaries.
    fn enforce_min_segment_size(
        &self,
        boundaries: &[usize],
        num_blocks: usize,
        min_blocks: usize,
    ) -> Vec<usize> {
        if boundaries.is_empty() || min_blocks == 0 {
            return boundaries.to_vec();
        }

        let mut filtered = vec![];
        let mut last_boundary = 0usize;

        for &boundary in boundaries {
            // Check distance from last boundary (or start)
            if boundary - last_boundary >= min_blocks {
                // Check distance to end
                if num_blocks - boundary >= min_blocks {
                    filtered.push(boundary);
                    last_boundary = boundary;
                }
            }
        }

        filtered
    }

    /// Creates segments from sentences and boundary indices.
    fn create_segments(
        &self,
        sentences: &[String],
        boundaries: &[usize],
    ) -> Vec<SemanticSegment> {
        let mut segments = vec![];
        let mut start = 0;

        for &boundary in boundaries {
            if boundary > start && boundary <= sentences.len() {
                let segment_sentences: Vec<String> =
                    sentences[start..boundary].to_vec();
                segments.push(SemanticSegment {
                    text: segment_sentences.join(" "),
                    sentences: segment_sentences,
                    start_idx: start,
                    end_idx: boundary,
                });
                start = boundary;
            }
        }

        // Add final segment
        if start < sentences.len() {
            let segment_sentences: Vec<String> = sentences[start..].to_vec();
            segments.push(SemanticSegment {
                text: segment_sentences.join(" "),
                sentences: segment_sentences,
                start_idx: start,
                end_idx: sentences.len(),
            });
        }

        segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depth_scores() {
        // Create a mock segmenter just to test the algorithm
        // In real tests, we'd need a proper retina
        let similarities = vec![0.8, 0.7, 0.3, 0.6, 0.9, 0.4, 0.8];

        // Manually compute expected depths
        // Position 2 (0.3) should have high depth because it dips below peaks
        let _config = SegmentationConfig::default();

        // Test that depth at local minima is higher
        let depths = compute_depth_scores_standalone(&similarities);

        // The depth at position 2 (similarity=0.3) should be significant
        // Left peak up to pos 2 is max(0.8, 0.7, 0.3) = 0.8
        // Right peak from pos 2 is max(0.3, 0.6, 0.9, 0.4, 0.8) = 0.9
        // Depth = (0.8 - 0.3) + (0.9 - 0.3) = 0.5 + 0.6 = 1.1
        assert!(depths[2] > depths[0]);
        assert!(depths[2] > depths[4]);
    }

    fn compute_depth_scores_standalone(similarities: &[f32]) -> Vec<f32> {
        let n = similarities.len();
        let mut depths = vec![0.0f32; n];

        for i in 0..n {
            let left_peak = similarities[..=i]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let right_peak = similarities[i..]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            depths[i] = (left_peak - similarities[i]) + (right_peak - similarities[i]);
        }

        depths
    }

    #[test]
    fn test_gaussian_smooth() {
        let scores = vec![0.0, 0.0, 1.0, 0.0, 0.0];

        // After smoothing, the peak should spread out
        let smoothed = gaussian_smooth_standalone(&scores, 1.0);

        // Center should still be highest but neighboring values should increase
        assert!(smoothed[2] > smoothed[0]);
        assert!(smoothed[1] > scores[1]);
        assert!(smoothed[3] > scores[3]);
    }

    fn gaussian_smooth_standalone(scores: &[f32], sigma: f32) -> Vec<f32> {
        let kernel_radius = (3.0 * sigma).ceil() as usize;
        let kernel_size = 2 * kernel_radius + 1;

        let mut kernel = vec![0.0f32; kernel_size];
        let sigma_sq = sigma * sigma;
        let mut sum = 0.0f32;

        for i in 0..kernel_size {
            let x = i as f32 - kernel_radius as f32;
            kernel[i] = (-x * x / (2.0 * sigma_sq)).exp();
            sum += kernel[i];
        }

        for k in &mut kernel {
            *k /= sum;
        }

        let mut smoothed = vec![0.0f32; scores.len()];
        for i in 0..scores.len() {
            let mut value = 0.0f32;
            for (j, &k) in kernel.iter().enumerate() {
                let idx = i as isize + j as isize - kernel_radius as isize;
                let clamped_idx = idx.clamp(0, scores.len() as isize - 1) as usize;
                value += k * scores[clamped_idx];
            }
            smoothed[i] = value;
        }

        smoothed
    }

    #[test]
    fn test_mean_std() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (mean, std) = mean_std_standalone(&values);

        assert!((mean - 5.0).abs() < 0.01);
        assert!((std - 2.0).abs() < 0.01);
    }

    fn mean_std_standalone(values: &[f32]) -> (f32, f32) {
        let n = values.len() as f32;
        let mean = values.iter().sum::<f32>() / n;
        let variance = values.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
        (mean, variance.sqrt())
    }
}
