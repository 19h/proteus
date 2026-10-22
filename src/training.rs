//! Corpus-to-retina training with explicit context-map and word-map contracts.

use crate::{
    FingerprintConfig, ProteusError, Result, Retina, Som, SomConfig, SomTrainer, TextConfig,
    Tokenizer, WordFingerprint,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::io::BufRead;

pub mod angular;
mod corpus;
mod features;
mod folding;
mod spectral;

/// Geometry used to encode a snippet before map training.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum FeatureModel {
    /// Sublinear term frequency times smoothed inverse document frequency.
    #[default]
    Tfidf,
    /// TF-IDF weighted composition of local positive-PMI word-context vectors.
    ContextPpmi,
}

/// Reduction applied to the word-context feature matrix.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProjectionMethod {
    /// Seeded Rademacher projection, or exact coordinates when dimension is zero.
    #[default]
    Random,
    /// Approximate dominant right subspace using orthogonal power iteration.
    Spectral,
}

/// Interpolation and background correction when folding word incidence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FoldingConfig {
    /// Gaussian standard deviation in cells. Zero preserves raw incidence.
    pub sigma: f64,
    /// Divide smoothed incidence by smoothed snippet occupancy to this power.
    pub background_exponent: f64,
}

impl FoldingConfig {
    /// Original observed-cell-only folding, also used for old model metadata.
    pub fn incidence() -> Self {
        Self {
            sigma: 0.0,
            background_exponent: 0.0,
        }
    }

    /// Local Gaussian interpolation with square-root occupancy correction.
    pub fn gaussian() -> Self {
        Self {
            sigma: 1.0,
            background_exponent: 0.5,
        }
    }
}

impl Default for FoldingConfig {
    fn default() -> Self {
        Self::incidence()
    }
}

fn default_context_window() -> usize {
    5
}
fn default_context_alpha() -> f64 {
    0.75
}

fn default_projection_iterations() -> usize {
    2
}

/// Interpretation of physical newlines in the input corpus.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DocumentMode {
    /// Each nonempty line is one document.
    Lines,
    /// Blank lines separate documents; wrapped lines are joined.
    Paragraphs,
}

/// Explicit choices for this implementation of the published folding architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Map dimensions, schedules and seed. `weight_dimension` is set from the corpus.
    pub som: SomConfig,
    /// Shared training and inference normalization.
    pub text: TextConfig,
    /// Maximum fingerprint density and aggregation policy.
    pub fingerprint: FingerprintConfig,
    /// Input record boundaries.
    pub document_mode: DocumentMode,
    /// Maximum normalized tokens in a nonoverlapping snippet.
    pub snippet_tokens: usize,
    /// Minimum corpus token count of a retained word.
    pub min_count: usize,
    /// Exclude ubiquitous words from TF-IDF features above this document fraction.
    /// Words remain in output fingerprints if they meet `min_count`.
    pub max_document_frequency: f64,
    /// Projection dimension; zero preserves exact feature coordinates.
    pub projection_dim: usize,
    /// Feature geometry; absent in legacy metadata means TF-IDF.
    #[serde(default)]
    pub feature_model: FeatureModel,
    /// Maximum token distance for local PPMI, without crossing snippets.
    #[serde(default = "default_context_window")]
    pub context_window: usize,
    /// Exponent smoothing the context marginal distribution for PPMI.
    #[serde(default = "default_context_alpha")]
    pub context_alpha: f64,
    /// Spatial evidence aggregation; absent in legacy metadata means incidence.
    #[serde(default = "FoldingConfig::incidence")]
    pub folding: FoldingConfig,
    /// Random projection or a learned word-context subspace.
    #[serde(default)]
    pub projection_method: ProjectionMethod,
    /// Orthogonal power iterations for spectral projection.
    #[serde(default = "default_projection_iterations")]
    pub projection_iterations: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            som: SomConfig {
                seed: Some(42),
                ..SomConfig::default()
            },
            text: TextConfig::default(),
            fingerprint: FingerprintConfig::default(),
            document_mode: DocumentMode::Paragraphs,
            snippet_tokens: 64,
            min_count: 2,
            max_document_frequency: 0.5,
            projection_dim: 0,
            feature_model: FeatureModel::Tfidf,
            context_window: default_context_window(),
            context_alpha: default_context_alpha(),
            folding: FoldingConfig::default(),
            projection_method: ProjectionMethod::Random,
            projection_iterations: default_projection_iterations(),
        }
    }
}

/// Statistics and provenance saved with the trained retina.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingReport {
    /// Algorithm contract version, independent of file format version.
    pub algorithm: String,
    /// SHA-256 of length-prefixed normalized snippets, including boundaries.
    pub corpus_sha256: String,
    /// Effective parameters including the resolved random seed and feature dimension.
    pub config: TrainingConfig,
    /// Number of extracted snippets before zero-feature filtering.
    pub snippets: usize,
    /// Snippets assigned in context-map mode; zero in word-map mode.
    pub mapped_snippets: usize,
    /// Retained output words.
    pub vocabulary: usize,
    /// Number of context features before optional projection.
    pub features: usize,
    /// Distinct cells occupied by final sample BMUs.
    pub occupied_cells: usize,
    /// Mean Euclidean distance from training vectors to their final prototypes.
    pub quantization_error: f64,
    /// Support and compression diagnostics; absent in older training reports.
    #[serde(default)]
    pub quality: Option<QualityDiagnostics>,
}

/// Observable evidence limits, distinct from labelled semantic accuracy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDiagnostics {
    /// Words observed in only one usable snippet, regardless of token repetition.
    pub single_context_words: usize,
    /// Number of output words minus the number of distinct bit patterns.
    pub duplicate_fingerprints: usize,
    /// Average output fingerprint cardinality.
    pub mean_active_bits: f64,
    /// Fraction of training vectors assigned to the busiest cell.
    pub maximum_cell_fraction: f64,
    /// exp(entropy) of the empirical sample-to-cell distribution, in cells.
    pub effective_occupied_cells: f64,
}

impl TrainingConfig {
    /// Validate user-controlled parameters before reading the corpus.
    pub fn validate(&self) -> Result<()> {
        crate::som::training::validate_config(&self.som)?;
        self.fingerprint.validate()?;
        if self.projection_method == ProjectionMethod::Spectral
            && (self.feature_model == FeatureModel::Tfidf
                || self.projection_dim == 0
                || self.projection_iterations == 0)
        {
            return Err(ProteusError::Config(
                "Spectral projection requires PPMI features, positive dimension and positive iterations".into(),
            ));
        }
        if self.context_window == 0
            || !self.context_alpha.is_finite()
            || self.context_alpha <= 0.0
            || self.context_alpha > 1.0
            || !self.folding.sigma.is_finite()
            || self.folding.sigma < 0.0
            || !self.folding.background_exponent.is_finite()
            || !(0.0..=1.0).contains(&self.folding.background_exponent)
        {
            return Err(ProteusError::Config(
                "Invalid context or folding parameters".into(),
            ));
        }
        if self.snippet_tokens == 0 || self.min_count == 0 {
            return Err(ProteusError::Config(
                "snippet_tokens and min_count must be positive".into(),
            ));
        }
        if !self.max_document_frequency.is_finite()
            || !(0.0 < self.max_document_frequency && self.max_document_frequency <= 1.0)
        {
            return Err(ProteusError::Config(
                "max_document_frequency must be in (0, 1]".into(),
            ));
        }
        if self.text.min_token_length > self.text.max_token_length {
            return Err(ProteusError::Config(
                "Minimum token length exceeds maximum".into(),
            ));
        }
        Ok(())
    }
}

/// Extract bounded snippets without interpreting hard-wrapped lines as sentences.
/// I/O and invalid UTF-8 errors propagate instead of silently truncating the corpus.
pub fn read_snippets(reader: impl BufRead, config: &TrainingConfig) -> Result<Vec<Vec<String>>> {
    config.validate()?;
    let tokenizer = Tokenizer::new(config.text.clone());
    let mut snippets = Vec::new();
    let mut pending = Vec::new();
    let flush = |pending: &mut Vec<String>, snippets: &mut Vec<Vec<String>>| {
        if !pending.is_empty() {
            snippets.push(std::mem::take(pending));
        }
    };
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            flush(&mut pending, &mut snippets);
            continue;
        }
        for token in tokenizer.tokenize_to_strings(&line) {
            pending.push(token);
            if pending.len() == config.snippet_tokens {
                flush(&mut pending, &mut snippets);
            }
        }
        if config.document_mode == DocumentMode::Lines {
            flush(&mut pending, &mut snippets);
        }
    }
    flush(&mut pending, &mut snippets);
    Ok(snippets)
}

/// Train a retina with exact vocabulary TF-IDF context folding by default.
/// A named alternative uses contextual PPMI features.
/// `progress` receives completed and total online updates.
pub fn train_corpus<F>(
    reader: impl BufRead,
    mut config: TrainingConfig,
    progress: F,
) -> Result<(Retina, TrainingReport)>
where
    F: Fn(usize, usize) + Sync + Send,
{
    config.validate()?;
    // A missing seed requests randomness, but the resolved seed is always persisted.
    let seed = config.som.seed.unwrap_or_else(|| rand::thread_rng().gen());
    config.som.seed = Some(seed);
    let corpus = corpus::PreparedCorpus::read(reader, &config)?;
    let snippets = &corpus.snippets;
    let counts = &corpus.word_counts;
    let features = corpus.feature_names(config.max_document_frequency)?;
    let idf = corpus.inverse_document_frequency(&features);
    let corpus_sha256 = corpus.digest.clone();
    let dimension = if config.projection_method == ProjectionMethod::Spectral {
        config.projection_dim.min(features.len())
    } else if config.projection_dim == 0 {
        features.len()
    } else {
        config.projection_dim
    };
    config.som.weight_dimension = dimension;
    let cells = config.som.total_neurons();
    cells
        .checked_mul(dimension)
        .and_then(|x| x.checked_mul(12))
        .ok_or_else(|| ProteusError::Config("Map allocation size overflow".into()))?;
    // A seeded dense Rademacher projection. Each feature row is generated once;
    // scaling 1/sqrt(d) preserves expected squared Euclidean lengths.
    let mut projection_rng = ChaCha8Rng::seed_from_u64(seed ^ 0x70726f6a656374);
    let projection: Vec<Vec<f32>> = if config.projection_dim == 0 {
        Vec::new()
    } else {
        let scale = 1.0 / (dimension as f32).sqrt();
        (0..features.len())
            .map(|_| {
                (0..dimension)
                    .map(|_| {
                        if projection_rng.gen::<bool>() {
                            scale
                        } else {
                            -scale
                        }
                    })
                    .collect()
            })
            .collect()
    };
    let feature_space =
        features::FeatureSpace::new(counts, &features, &idf, &projection, dimension);
    let encoded_snippets = feature_space.encode_snippets(snippets, &config);
    let inputs = encoded_snippets.vectors;
    let kept = encoded_snippets.source_indices;
    if inputs.is_empty() {
        return Err(ProteusError::Training("No nonzero snippet vectors".into()));
    }
    // Sample-based initialization keeps prototype scale consistent with input scale.
    let mut som = Som::new_zeros(config.som.dimension, dimension, config.som.toroidal);
    let mut init_rng = ChaCha8Rng::seed_from_u64(seed ^ 0x696e697469616c);
    for neuron in &mut som.neurons {
        let input = &inputs[init_rng.gen_range(0..inputs.len())];
        for (w, &x) in neuron.weights.iter_mut().zip(input) {
            *w = x as f64;
        }
    }
    let mut trainer = SomTrainer::new(config.som.clone());
    let bmus = trainer.train_vectors(&mut som, &inputs, progress)?;
    let mut hits = BTreeMap::<String, BTreeMap<usize, usize>>::new();
    let mut occupied = std::collections::BTreeSet::new();
    let mut occupancy = vec![0usize; cells];
    let mut error = 0.0;
    for ((&snippet_idx, &bmu), input) in kept.iter().zip(&bmus).zip(&inputs) {
        occupied.insert(bmu);
        occupancy[bmu] += 1;
        error += som.neurons[bmu]
            .weights
            .iter()
            .zip(input)
            .map(|(&w, &x)| (w - x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        // Incidence: one vote per distinct word per snippet, never token count.
        let terms: std::collections::BTreeSet<_> = snippets[snippet_idx].iter().collect();
        for word in terms {
            if counts.contains_key(word) {
                *hits
                    .entry(word.clone())
                    .or_default()
                    .entry(bmu)
                    .or_default() += 1;
            }
        }
    }
    let total_tokens: usize = counts.values().sum();
    let limit = config.fingerprint.active_bit_limit(cells as u32);
    let single_context_words = hits
        .values()
        .filter(|row| row.values().sum::<usize>() == 1)
        .count();
    let mut fingerprints = HashMap::new();
    let folder = folding::SpatialFolder::new(
        &occupancy,
        config.som.dimension,
        config.som.toroidal,
        &config.folding,
    );
    for (word, row) in hits {
        let positions = folder.fold(&row, limit);
        let frequency = counts[&word] as f64 / total_tokens as f64;
        fingerprints.insert(
            word.clone(),
            WordFingerprint::with_frequency(word, &positions, cells as u32, frequency),
        );
    }
    let mut patterns = std::collections::BTreeSet::new();
    let mut active_bits = 0usize;
    for fp in fingerprints.values() {
        active_bits += fp.cardinality() as usize;
        patterns.insert(fp.fingerprint.to_positions());
    }
    let entropy = occupancy
        .iter()
        .filter(|&&n| n > 0)
        .map(|&n| {
            let p = n as f64 / inputs.len() as f64;
            -p * p.ln()
        })
        .sum::<f64>();
    let quality = QualityDiagnostics {
        single_context_words,
        duplicate_fingerprints: fingerprints.len() - patterns.len(),
        mean_active_bits: active_bits as f64 / fingerprints.len() as f64,
        maximum_cell_fraction: *occupancy.iter().max().unwrap() as f64 / inputs.len() as f64,
        effective_occupied_cells: entropy.exp(),
    };
    let report = TrainingReport {
        algorithm: if config.folding.sigma == 0.0 && config.feature_model == FeatureModel::Tfidf {
            "semantic-folding-tfidf-online-som-v1".into()
        } else {
            "semantic-folding-context-online-som-v2".into()
        },
        corpus_sha256,
        snippets: snippets.len(),
        mapped_snippets: inputs.len(),
        vocabulary: fingerprints.len(),
        features: features.len(),
        occupied_cells: occupied.len(),
        quantization_error: error / inputs.len() as f64,
        quality: Some(quality),
        config: config.clone(),
    };
    let mut retina = Retina::with_index(
        fingerprints,
        config.som.dimension as u32,
        config.fingerprint.clone(),
    );
    retina.set_training_report(report.clone());
    Ok((retina, report))
}
