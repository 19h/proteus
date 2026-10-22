//! Corpus-to-retina training: TF-IDF snippets -> SOM -> word/context incidence.

use crate::som::simd::normalize_f32;
use crate::{
    FingerprintConfig, ProteusError, Result, Retina, Som, SomConfig, SomTrainer, TextConfig,
    Tokenizer, WordFingerprint,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::io::BufRead;

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
    /// Optional random projection dimension; zero uses exact vocabulary TF-IDF.
    pub projection_dim: usize,
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
    /// Snippets that supplied finite nonzero vectors.
    pub mapped_snippets: usize,
    /// Retained output words.
    pub vocabulary: usize,
    /// Number of vocabulary TF-IDF features before optional projection.
    pub features: usize,
    /// Distinct cells occupied by final snippet BMUs.
    pub occupied_cells: usize,
    /// Mean Euclidean distance from snippet vectors to final prototypes.
    pub quantization_error: f64,
}

impl TrainingConfig {
    /// Validate user-controlled parameters before reading the corpus.
    pub fn validate(&self) -> Result<()> {
        crate::som::training::validate_config(&self.som)?;
        self.fingerprint.validate()?;
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

/// Train a retina with exact vocabulary TF-IDF by default. Projection is opt-in.
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
    let snippets = read_snippets(reader, &config)?;
    if snippets.is_empty() {
        return Err(ProteusError::EmptyInput(
            "No normalized corpus tokens".into(),
        ));
    }
    let mut digest = Sha256::new();
    for snippet in &snippets {
        digest.update((snippet.len() as u64).to_le_bytes());
        for token in snippet {
            digest.update((token.len() as u64).to_le_bytes());
            digest.update(token.as_bytes());
        }
    }
    let corpus_sha256 = format!("{:x}", digest.finalize());
    let mut counts = BTreeMap::<String, usize>::new();
    let mut df = BTreeMap::<String, usize>::new();
    for snippet in &snippets {
        let mut terms = BTreeMap::<&str, usize>::new();
        for word in snippet {
            *counts.entry(word.clone()).or_default() += 1;
            *terms.entry(word).or_default() += 1;
        }
        for word in terms.keys() {
            *df.entry((*word).to_owned()).or_default() += 1;
        }
    }
    counts.retain(|_, n| *n >= config.min_count);
    if counts.is_empty() {
        return Err(ProteusError::Training("No words meet min_count".into()));
    }
    let features: Vec<&str> = counts
        .keys()
        .filter(|w| df[*w] as f64 / snippets.len() as f64 <= config.max_document_frequency)
        .map(String::as_str)
        .collect();
    if features.is_empty() {
        return Err(ProteusError::Training("No TF-IDF features remain; increase max_document_frequency or supply more diverse contexts".into()));
    }
    let feature_ids: HashMap<&str, usize> =
        features.iter().enumerate().map(|(i, &w)| (w, i)).collect();
    let idf: Vec<f32> = features
        .iter()
        .map(|w| ((1.0 + snippets.len() as f64) / (1.0 + df[*w] as f64)).ln() as f32 + 1.0)
        .collect();
    let dimension = if config.projection_dim == 0 {
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
    let mut inputs = Vec::new();
    let mut kept = Vec::new();
    for (i, snippet) in snippets.iter().enumerate() {
        let mut tf = BTreeMap::<usize, usize>::new();
        for word in snippet {
            if let Some(&j) = feature_ids.get(word.as_str()) {
                *tf.entry(j).or_default() += 1;
            }
        }
        if tf.is_empty() {
            continue;
        }
        let mut v = vec![0.0; dimension];
        for (j, count) in tf {
            let weight = (1.0 + (count as f32).ln()) * idf[j];
            if projection.is_empty() {
                v[j] = weight;
            } else {
                for (out, &r) in v.iter_mut().zip(&projection[j]) {
                    *out += weight * r;
                }
            }
        }
        if v.iter().all(|&x| x == 0.0) {
            continue;
        }
        normalize_f32(&mut v);
        inputs.push(v);
        kept.push(i);
    }
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
    let mut error = 0.0;
    for ((&snippet_idx, &bmu), input) in kept.iter().zip(&bmus).zip(&inputs) {
        occupied.insert(bmu);
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
    let mut fingerprints = HashMap::new();
    for (word, row) in hits {
        let mut row: Vec<_> = row.into_iter().collect();
        row.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let positions: Vec<u32> = row
            .into_iter()
            .take(limit)
            .map(|(cell, _)| cell as u32)
            .collect();
        let frequency = counts[&word] as f64 / total_tokens as f64;
        fingerprints.insert(
            word.clone(),
            WordFingerprint::with_frequency(word, &positions, cells as u32, frequency),
        );
    }
    let report = TrainingReport {
        algorithm: "semantic-folding-tfidf-online-som-v1".into(),
        corpus_sha256,
        snippets: snippets.len(),
        mapped_snippets: inputs.len(),
        vocabulary: fingerprints.len(),
        features: features.len(),
        occupied_cells: occupied.len(),
        quantization_error: error / inputs.len() as f64,
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
