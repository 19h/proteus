//! Deterministic lexical and local word-context feature encoders.

use super::{spectral, FeatureModel, ProjectionMethod, TrainingConfig};
use crate::som::simd::normalize_f32;
use std::collections::{BTreeMap, HashMap};

/// Shared vocabulary and projection coordinates for one training corpus.
pub(super) struct FeatureSpace<'a> {
    feature_indices: HashMap<&'a str, usize>,
    word_indices: HashMap<&'a str, usize>,
    words: Vec<&'a str>,
    inverse_document_frequency: &'a [f32],
    projection: &'a [Vec<f32>],
    dimension: usize,
}

/// Nonzero vectors and their corresponding source snippet indices.
pub(super) struct EncodedSnippets {
    pub vectors: Vec<Vec<f32>>,
    pub source_indices: Vec<usize>,
}

impl<'a> FeatureSpace<'a> {
    pub(super) fn new(
        word_counts: &'a BTreeMap<String, usize>,
        feature_names: &[&'a str],
        inverse_document_frequency: &'a [f32],
        projection: &'a [Vec<f32>],
        dimension: usize,
    ) -> Self {
        let feature_indices = feature_names
            .iter()
            .enumerate()
            .map(|(index, &word)| (word, index))
            .collect();
        let words: Vec<&str> = word_counts.keys().map(String::as_str).collect();
        let word_indices = words
            .iter()
            .enumerate()
            .map(|(index, &word)| (word, index))
            .collect();

        Self {
            feature_indices,
            word_indices,
            words,
            inverse_document_frequency,
            projection,
            dimension,
        }
    }

    pub(super) fn encode_snippets(
        &self,
        snippets: &[Vec<String>],
        config: &TrainingConfig,
    ) -> EncodedSnippets {
        let context_vectors = match config.feature_model {
            FeatureModel::Tfidf => Vec::new(),
            FeatureModel::ContextPpmi => self.encode_word_contexts(snippets, config, false),
        };
        let term_indices = match config.feature_model {
            FeatureModel::Tfidf => &self.feature_indices,
            FeatureModel::ContextPpmi => &self.word_indices,
        };
        let mut encoded = EncodedSnippets {
            vectors: Vec::with_capacity(snippets.len()),
            source_indices: Vec::with_capacity(snippets.len()),
        };

        for (snippet_index, snippet) in snippets.iter().enumerate() {
            let mut term_counts = BTreeMap::<usize, usize>::new();
            for word in snippet {
                if let Some(&term_index) = term_indices.get(word.as_str()) {
                    *term_counts.entry(term_index).or_default() += 1;
                }
            }

            let mut snippet_vector = vec![0.0; self.dimension];
            for (term_index, count) in term_counts {
                let term_frequency = 1.0 + (count as f32).ln();
                match config.feature_model {
                    FeatureModel::Tfidf => {
                        let weight = term_frequency * self.inverse_document_frequency[term_index];
                        self.add_projected_feature(&mut snippet_vector, term_index, weight);
                    }
                    FeatureModel::ContextPpmi => {
                        let word = self.words[term_index];
                        let inverse_document_frequency = self
                            .feature_indices
                            .get(word)
                            .map_or(0.0, |&index| self.inverse_document_frequency[index]);
                        let weight = term_frequency * inverse_document_frequency;
                        for (coordinate, &context_value) in
                            snippet_vector.iter_mut().zip(&context_vectors[term_index])
                        {
                            *coordinate += weight * context_value;
                        }
                    }
                }
            }

            if snippet_vector.iter().all(|&coordinate| coordinate == 0.0) {
                continue;
            }
            normalize_f32(&mut snippet_vector);
            encoded.vectors.push(snippet_vector);
            encoded.source_indices.push(snippet_index);
        }

        encoded
    }

    pub(super) fn encode_word_contexts(
        &self,
        snippets: &[Vec<String>],
        config: &TrainingConfig,
        whiten: bool,
    ) -> Vec<Vec<f32>> {
        let mut cooccurrences = vec![BTreeMap::<usize, f64>::new(); self.words.len()];
        let mut context_totals = vec![0.0_f64; self.feature_indices.len()];
        let window_radius = config.context_window;

        for snippet in snippets {
            for (word_position, word) in snippet.iter().enumerate() {
                let Some(&word_index) = self.word_indices.get(word.as_str()) else {
                    continue;
                };

                // Filtering never shortens the original token distances.
                let start = word_position.saturating_sub(window_radius);
                let end = word_position
                    .saturating_add(window_radius)
                    .saturating_add(1)
                    .min(snippet.len());
                for (context_position, context) in snippet.iter().enumerate().take(end).skip(start)
                {
                    if context_position == word_position {
                        continue;
                    }
                    let Some(&context_index) = self.feature_indices.get(context.as_str()) else {
                        continue;
                    };

                    // Expected inclusion under a uniformly sampled window radius.
                    let distance = word_position.abs_diff(context_position);
                    let weight = (window_radius - distance + 1) as f64 / window_radius as f64;
                    *cooccurrences[word_index].entry(context_index).or_default() += weight;
                    context_totals[context_index] += weight;
                }
            }
        }

        let smoothed_context_totals: Vec<f64> = context_totals
            .iter()
            .map(|total| total.powf(config.context_alpha))
            .collect();
        let context_normalizer: f64 = smoothed_context_totals.iter().sum();

        let positive_pmi_rows: Vec<spectral::SparseRow> = cooccurrences
            .iter()
            .map(|context_counts| {
                let word_total: f64 = context_counts.values().sum();
                context_counts
                    .iter()
                    .filter_map(|(&context_index, &count)| {
                        let conditional_probability = count / word_total;
                        let context_probability =
                            smoothed_context_totals[context_index] / context_normalizer;
                        let positive_pmi = (conditional_probability / context_probability)
                            .ln()
                            .max(0.0) as f32;
                        (positive_pmi > 0.0).then_some((context_index, positive_pmi))
                    })
                    .collect()
            })
            .collect();

        let mut vectors = if config.projection_method == ProjectionMethod::Spectral {
            spectral::project(
                &positive_pmi_rows,
                self.projection,
                config.projection_iterations,
            )
        } else {
            positive_pmi_rows
                .iter()
                .map(|row| {
                    let mut vector = vec![0.0; self.dimension];
                    for &(feature, value) in row {
                        self.add_projected_feature(&mut vector, feature, value);
                    }
                    vector
                })
                .collect()
        };
        if whiten {
            spectral::whiten_rows(&mut vectors);
        }
        for vector in &mut vectors {
            normalize_f32(vector);
        }
        vectors
    }

    pub(super) fn word(&self, index: usize) -> &str {
        self.words[index]
    }

    fn add_projected_feature(&self, output: &mut [f32], feature_index: usize, weight: f32) {
        if self.projection.is_empty() {
            output[feature_index] += weight;
            return;
        }

        for (coordinate, &projection_value) in
            output.iter_mut().zip(&self.projection[feature_index])
        {
            *coordinate += weight * projection_value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_ppmi_encodes_substitution_without_word_cooccurrence() {
        // "king" and "queen" never occur together, but share the context "rules".
        // Every observed association has PPMI = ln(2) for this balanced corpus.
        let snippets = vec![
            vec!["king".into(), "rules".into()],
            vec!["queen".into(), "rules".into()],
        ];
        let word_counts =
            BTreeMap::from([("king".into(), 1), ("queen".into(), 1), ("rules".into(), 2)]);
        let feature_names = ["king", "queen", "rules"];
        let inverse_document_frequency = [(1.5_f32).ln() + 1.0, (1.5_f32).ln() + 1.0, 1.0];
        let feature_space = FeatureSpace::new(
            &word_counts,
            &feature_names,
            &inverse_document_frequency,
            &[],
            feature_names.len(),
        );
        let config = TrainingConfig {
            feature_model: FeatureModel::ContextPpmi,
            context_window: 1,
            context_alpha: 1.0,
            ..TrainingConfig::default()
        };

        let encoded = feature_space.encode_snippets(&snippets, &config);
        assert_eq!(encoded.source_indices, vec![0, 1]);
        assert_eq!(encoded.vectors[0], encoded.vectors[1]);

        // The word-context vectors are (0, 0, 1) and (1/sqrt(2), 1/sqrt(2), 0).
        // Compose them with TF-IDF, then normalize independently of the encoder.
        let mut expected_vector = [
            1.0 / 2.0_f32.sqrt(),
            1.0 / 2.0_f32.sqrt(),
            inverse_document_frequency[0],
        ];
        let expected_norm = expected_vector
            .iter()
            .map(|coordinate| coordinate * coordinate)
            .sum::<f32>()
            .sqrt();
        for coordinate in &mut expected_vector {
            *coordinate /= expected_norm;
        }
        for (actual, expected) in encoded.vectors[0].iter().zip(expected_vector) {
            assert!((actual - expected).abs() < 2e-7);
        }
    }

    #[test]
    fn local_contexts_do_not_cross_snippet_boundaries() {
        let snippets = vec![
            vec!["king".into()],
            vec!["queen".into()],
            vec!["rules".into()],
        ];
        let word_counts =
            BTreeMap::from([("king".into(), 1), ("queen".into(), 1), ("rules".into(), 1)]);
        let feature_names = ["king", "queen", "rules"];
        let inverse_document_frequency = [1.0; 3];
        let feature_space = FeatureSpace::new(
            &word_counts,
            &feature_names,
            &inverse_document_frequency,
            &[],
            feature_names.len(),
        );
        let config = TrainingConfig {
            feature_model: FeatureModel::ContextPpmi,
            ..TrainingConfig::default()
        };

        let encoded = feature_space.encode_snippets(&snippets, &config);

        assert!(encoded.vectors.is_empty());
        assert!(encoded.source_indices.is_empty());
    }
}
