//! Distributional word vectors encoded directly as sparse angular fingerprints.
//!
//! Bit positions are random feature identifiers. They have no learned spatial
//! adjacency, unlike the cells of the reference self-organizing map.

use super::corpus::PreparedCorpus;
use super::features::FeatureSpace;
use super::{DocumentMode, FeatureModel, ProjectionMethod, TrainingConfig};
use crate::som::simd::dot_product_f32;
use crate::{
    FingerprintConfig, ProteusError, Result, Retina, SomConfig, TextConfig, WordFingerprint,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::io::BufRead;

/// Configuration for the non-spatial distributional training backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AngularConfig {
    /// Square display dimension; its square is the number of random features.
    pub dimension: usize,
    /// Shared training and inference normalization.
    pub text: TextConfig,
    /// Binary density cap and text-composition policy.
    pub fingerprint: FingerprintConfig,
    /// Input record boundaries.
    pub document_mode: DocumentMode,
    /// Maximum tokens per context snippet.
    pub snippet_tokens: usize,
    /// Minimum retained word frequency.
    pub min_count: usize,
    /// Maximum snippet frequency of a context feature.
    pub max_document_frequency: f64,
    /// Maximum dimension of the learned word-context subspace.
    pub embedding_dimension: usize,
    /// Orthogonal subspace iterations.
    pub power_iterations: usize,
    /// Equalize the learned spectral directions before normalizing word vectors.
    /// Missing legacy metadata means false; new training defaults to true.
    #[serde(default)]
    pub whiten: bool,
    /// Maximum word-context token distance.
    pub context_window: usize,
    /// Context marginal smoothing exponent.
    pub context_alpha: f64,
    /// Seed for independent subspace and angular-projection random streams.
    pub seed: u64,
    /// Worker count; zero selects the Rayon default.
    pub threads: usize,
}

impl Default for AngularConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            text: TextConfig::default(),
            fingerprint: FingerprintConfig::default(),
            document_mode: DocumentMode::Paragraphs,
            snippet_tokens: 64,
            min_count: 2,
            max_document_frequency: 0.5,
            embedding_dimension: 128,
            power_iterations: 2,
            whiten: true,
            context_window: 5,
            context_alpha: 0.75,
            seed: 42,
            threads: 0,
        }
    }
}

impl AngularConfig {
    /// Validate dimensions, density and corpus-feature settings before reading input.
    pub fn validate(&self) -> Result<()> {
        if self.dimension == 0
            || self
                .dimension
                .checked_mul(self.dimension)
                .is_none_or(|cells| cells > u32::MAX as usize)
        {
            return Err(ProteusError::Config(
                "Angular display dimension must be positive and its square fit in u32".into(),
            ));
        }
        if self.embedding_dimension == 0 || self.power_iterations == 0 {
            return Err(ProteusError::Config(
                "embedding_dimension and power_iterations must be positive".into(),
            ));
        }
        self.feature_config().validate()
    }

    fn feature_config(&self) -> TrainingConfig {
        // The shared feature builder also accepts map configuration. These map
        // fields are internal validation scaffolding; no SOM is trained here.
        TrainingConfig {
            som: SomConfig {
                dimension: self.dimension,
                iterations: 1,
                seed: Some(self.seed),
                ..SomConfig::default()
            },
            text: self.text.clone(),
            fingerprint: self.fingerprint.clone(),
            document_mode: self.document_mode,
            snippet_tokens: self.snippet_tokens,
            min_count: self.min_count,
            max_document_frequency: self.max_document_frequency,
            projection_dim: self.embedding_dimension,
            feature_model: FeatureModel::ContextPpmi,
            projection_method: ProjectionMethod::Spectral,
            projection_iterations: self.power_iterations,
            context_window: self.context_window,
            context_alpha: self.context_alpha,
            ..TrainingConfig::default()
        }
    }
}

/// Reproducible provenance for an angular retina, separate from SOM training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AngularReport {
    /// Encoding and feature-pipeline contract identifier.
    pub algorithm: String,
    /// Effective configuration, including the seed and density cap.
    pub config: AngularConfig,
    /// Hash of the length-prefixed normalized snippets.
    pub corpus_sha256: String,
    /// Extracted source snippets.
    pub snippets: usize,
    /// Context features before subspace projection.
    pub features: usize,
    /// Actual vector dimension, bounded by feature count.
    pub vector_dimension: usize,
    /// Encoded words.
    pub vocabulary: usize,
    /// Retained words without a nonzero context vector.
    pub zero_context_words: usize,
    /// Active bits in every output word fingerprint.
    pub active_bits: usize,
    /// Output words minus distinct binary patterns.
    pub duplicate_fingerprints: usize,
}

/// Train PPMI word vectors and encode their Gaussian projection order statistics.
/// Progress counts encoded words, not SOM updates. No map optimizer is invoked.
pub fn train_distributional<F>(
    reader: impl BufRead + Send,
    config: AngularConfig,
    progress: F,
) -> Result<(Retina, AngularReport)>
where
    F: Fn(usize, usize) + Send + Sync,
{
    config.validate()?;
    if config.threads == 0 {
        train_inner(reader, config, progress)
    } else {
        rayon::ThreadPoolBuilder::new()
            .num_threads(config.threads)
            .build()
            .map_err(|error| ProteusError::Training(error.to_string()))?
            .install(|| train_inner(reader, config, progress))
    }
}

fn train_inner<F>(
    reader: impl BufRead,
    config: AngularConfig,
    progress: F,
) -> Result<(Retina, AngularReport)>
where
    F: Fn(usize, usize) + Send + Sync,
{
    let feature_config = config.feature_config();
    let corpus = PreparedCorpus::read(reader, &feature_config)?;
    let features = corpus.feature_names(config.max_document_frequency)?;
    let dimension = config.embedding_dimension.min(features.len());
    let inverse_document_frequency = corpus.inverse_document_frequency(&features);
    let mut subspace_generator = ChaCha8Rng::seed_from_u64(config.seed ^ 0x70726f6a656374);
    let projection: Vec<Vec<f32>> = (0..features.len())
        .map(|_| {
            (0..dimension)
                .map(|_| {
                    if subspace_generator.gen::<bool>() {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect()
        })
        .collect();
    let feature_space = FeatureSpace::new(
        &corpus.word_counts,
        &features,
        &inverse_document_frequency,
        &projection,
        dimension,
    );
    let word_vectors =
        feature_space.encode_word_contexts(&corpus.snippets, &feature_config, config.whiten);
    let words: Vec<_> = word_vectors
        .into_iter()
        .enumerate()
        .filter(|(_, vector)| vector.iter().any(|&coordinate| coordinate != 0.0))
        .map(|(index, vector)| (feature_space.word(index).to_owned(), vector))
        .collect();
    if words.is_empty() {
        return Err(ProteusError::Training(
            "No nonzero word-context vectors".into(),
        ));
    }
    let feature_count = features.len();
    drop(feature_space);
    drop(projection);

    let feature_slots = config.dimension * config.dimension;
    feature_slots
        .checked_mul(dimension)
        .and_then(|size| size.checked_mul(4))
        .ok_or_else(|| ProteusError::Config("Angular projection allocation overflow".into()))?;
    let mut angular_generator = ChaCha8Rng::seed_from_u64(config.seed ^ 0x616e67756c6172);
    let directions: Vec<Vec<f32>> = (0..feature_slots)
        .map(|_| {
            (0..dimension)
                .map(|_| angular_generator.sample(StandardNormal))
                .collect()
        })
        .collect();
    let active_bits = config.fingerprint.active_bit_limit(feature_slots as u32);
    let token_count: usize = corpus.word_counts.values().sum();
    let mut fingerprints = HashMap::new();
    for chunk in words.chunks(128) {
        let encoded: Vec<_> = chunk
            .par_iter()
            .map(|(word, vector)| {
                let positions = encode(vector, &directions, active_bits);
                let frequency = corpus.word_counts[word] as f64 / token_count as f64;
                let fingerprint = WordFingerprint::with_frequency(
                    word.clone(),
                    &positions,
                    feature_slots as u32,
                    frequency,
                );
                (word.clone(), fingerprint)
            })
            .collect();
        fingerprints.extend(encoded);
        progress(fingerprints.len(), words.len());
    }
    let distinct_patterns: BTreeSet<_> = fingerprints
        .values()
        .map(|fingerprint| fingerprint.fingerprint.to_positions())
        .collect();
    let report = AngularReport {
        algorithm: "distributional-ppmi-spectral-angular-v2".into(),
        config: config.clone(),
        corpus_sha256: corpus.digest,
        snippets: corpus.snippets.len(),
        features: feature_count,
        vector_dimension: dimension,
        vocabulary: fingerprints.len(),
        zero_context_words: corpus.word_counts.len() - fingerprints.len(),
        active_bits,
        duplicate_fingerprints: fingerprints.len() - distinct_patterns.len(),
    };
    let mut retina = Retina::with_index(fingerprints, config.dimension as u32, config.fingerprint);
    retina.set_angular_report(report.clone());
    Ok((retina, report))
}

fn encode(vector: &[f32], directions: &[Vec<f32>], active_bits: usize) -> Vec<u32> {
    let mut scores: Vec<_> = directions
        .iter()
        .enumerate()
        .map(|(position, direction)| (position, dot_product_f32(vector, direction)))
        .collect();
    if scores.len() > active_bits {
        scores.select_nth_unstable_by(active_bits, |left, right| {
            right.1.total_cmp(&left.1).then(left.0.cmp(&right.0))
        });
        scores.truncate(active_bits);
    }
    scores
        .into_iter()
        .map(|(position, _)| position as u32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_statistics_select_the_strongest_features_with_stable_ties() {
        let directions = vec![
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
        ];
        let mut positions = encode(&[1.0, 0.0], &directions, 2);
        positions.sort_unstable();
        assert_eq!(positions, vec![0, 1]);
        assert_eq!(encode(&[1.0, 0.0], &directions, 1), vec![0]);
    }
}

#[cfg(test)]
mod angular_properties {
    use super::*;

    #[test]
    fn fixed_gaussian_features_preserve_well_separated_angles() {
        let mut generator = ChaCha8Rng::seed_from_u64(42);
        let directions: Vec<Vec<f32>> = (0..16384)
            .map(|_| (0..2).map(|_| generator.sample(StandardNormal)).collect())
            .collect();
        let signature = |vector: &[f32]| -> BTreeSet<u32> {
            encode(vector, &directions, 328).into_iter().collect()
        };
        let reference = signature(&[1.0, 0.0]);
        let nearby = signature(&[0.8, 0.6]);
        let orthogonal = signature(&[0.0, 1.0]);
        let opposite = signature(&[-1.0, 0.0]);

        assert_eq!(reference, signature(&[2.0, 0.0]));
        assert_eq!(reference.len(), 328);
        assert!(
            reference.intersection(&nearby).count() > reference.intersection(&orthogonal).count()
        );
        assert_eq!(reference.intersection(&opposite).count(), 0);
    }
}
