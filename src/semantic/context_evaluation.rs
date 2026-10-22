//! Complete-observation contextual evaluation with explicit missing/error coverage.

use super::{ContextualWord, SemanticModel, WordQuery};
use crate::{ProteusError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::io::BufRead;

/// One contextual comparison, optionally carrying a same-meaning reference label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPair {
    /// Unique observation identity; duplicate word pairs remain separate observations.
    pub id: String,
    /// First target usage.
    pub left: ContextualWord,
    /// Second target usage.
    pub right: ContextualWord,
    /// Optional gold same-meaning label.
    #[serde(default)]
    pub same_sense: Option<bool>,
}

/// Coverage and accuracy for one explicitly named contextual classifier.
#[derive(Debug, Clone, Serialize)]
pub struct ContextMetrics {
    /// All pairs with an available prediction, including unlabelled observations.
    pub available_pairs: usize,
    /// Fraction of all input pairs with an available prediction.
    pub coverage: f64,
    /// Labelled observations with an available prediction.
    pub labelled_available: usize,
    /// Correct covered labelled observations.
    pub correct: usize,
    /// Accuracy on labelled available pairs, or None without such observations.
    pub accuracy: Option<f64>,
    /// Correct / all labelled pairs, counting unavailable predictions as errors.
    pub accuracy_with_unavailable_as_errors: Option<f64>,
}

/// Retained prediction and failures for an individual input observation.
#[derive(Debug, Clone, Serialize)]
pub struct ContextPairPrediction {
    /// Original observation identity.
    pub id: String,
    /// Original optional label.
    pub gold: Option<bool>,
    /// Legacy MAP equality without POS filtering or target exclusion.
    pub baseline: Option<bool>,
    /// Target-aware MAP equality.
    pub predicted: Option<bool>,
    /// Legacy selected sense identifiers, in left/right order.
    pub baseline_senses: [Option<String>; 2],
    /// Target-aware selected sense identifiers, in left/right order.
    pub selected_senses: [Option<String>; 2],
    /// Explicit unavailable reasons or invalid-query errors, retained per observation.
    pub errors: Vec<String>,
}

/// Paired contextual evaluation. Accuracy is an empirical label metric, not confidence.
#[derive(Debug, Clone, Serialize)]
pub struct ContextEvaluation {
    /// Candidate inference contract.
    pub algorithm: &'static str,
    /// Canonical annotated-corpus identity in the loaded model.
    pub corpus_sha256: String,
    /// Canonical lexical-resource identity in the loaded model.
    pub lexicon_sha256: String,
    /// All supplied observations, including missing and invalid target queries.
    pub total_pairs: usize,
    /// Observations with labels.
    pub labelled_pairs: usize,
    /// Legacy gloss/POS-unrestricted baseline.
    pub baseline: ContextMetrics,
    /// Target-aware candidate.
    pub candidate: ContextMetrics,
    /// Labelled observations covered by both methods.
    pub common_labelled_pairs: usize,
    /// Baseline accuracy on exactly the common labelled observations.
    pub common_baseline_accuracy: Option<f64>,
    /// Candidate accuracy on exactly the common labelled observations.
    pub common_candidate_accuracy: Option<f64>,
    /// Baseline incorrect, candidate correct, among common labelled observations.
    pub improvements: usize,
    /// Baseline correct, candidate incorrect, among common labelled observations.
    pub regressions: usize,
    /// Every input observation in its original order.
    pub predictions: Vec<ContextPairPrediction>,
}

/// Parse JSONL contextual pairs, rejecting malformed records and duplicate IDs.
/// No record is silently dropped except blank lines.
pub fn read_context_pairs(reader: impl BufRead) -> Result<Vec<ContextPair>> {
    let mut pairs = Vec::new();
    let mut ids = BTreeSet::new();
    for (line_number, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let pair: ContextPair = serde_json::from_str(&line).map_err(|error| {
            ProteusError::Config(format!(
                "Invalid contextual pair at line {}: {error}",
                line_number + 1
            ))
        })?;
        if pair.id.is_empty() || !ids.insert(pair.id.clone()) {
            return Err(ProteusError::Config(
                "Contextual observation IDs must be nonempty and unique".into(),
            ));
        }
        pairs.push(pair);
    }
    Ok(pairs)
}

/// Evaluate legacy and target-aware sense equality in one model load. Invalid
/// target queries remain unavailable observations; model/resource errors abort.
pub fn evaluate_context_pairs(
    model: &SemanticModel,
    pairs: &[ContextPair],
) -> Result<ContextEvaluation> {
    let mut predictions = Vec::with_capacity(pairs.len());
    for pair in pairs {
        let mut baseline_senses = [None, None];
        let mut selected_senses = [None, None];
        let mut errors = Vec::new();
        for (side, query) in [&pair.left, &pair.right].into_iter().enumerate() {
            match model.explain_senses(WordQuery {
                word: &query.word,
                context: Some(&query.context),
                sense: None,
            }) {
                Ok(senses) => baseline_senses[side] = senses.first().map(|sense| sense.id.clone()),
                Err(ProteusError::WordNotFound(_)) => {}
                Err(error) => return Err(error),
            }
            match model.disambiguate(query) {
                Ok(result) => {
                    selected_senses[side] = result.selected_sense;
                    if let Some(reason) = result.unavailable {
                        errors.push(format!("side {side}: {reason}"));
                    }
                }
                Err(ProteusError::Config(message)) => {
                    errors.push(format!("side {side}: {message}"))
                }
                Err(error) => return Err(error),
            }
        }
        let equality = |senses: &[Option<String>; 2]| {
            senses[0]
                .as_ref()
                .zip(senses[1].as_ref())
                .map(|(left, right)| left == right)
        };
        predictions.push(ContextPairPrediction {
            id: pair.id.clone(),
            gold: pair.same_sense,
            baseline: equality(&baseline_senses),
            predicted: equality(&selected_senses),
            baseline_senses,
            selected_senses,
            errors,
        });
    }
    let labelled_pairs = predictions.iter().filter(|row| row.gold.is_some()).count();
    let common: Vec<_> = predictions
        .iter()
        .filter(|row| row.gold.is_some() && row.baseline.is_some() && row.predicted.is_some())
        .collect();
    let common_baseline = common.iter().filter(|row| row.baseline == row.gold).count();
    let common_candidate = common
        .iter()
        .filter(|row| row.predicted == row.gold)
        .count();
    Ok(ContextEvaluation {
        algorithm: "target-gloss-v1",
        corpus_sha256: model.report().corpus_sha256.clone(),
        lexicon_sha256: model.report().lexicon_sha256.clone(),
        total_pairs: pairs.len(),
        labelled_pairs,
        baseline: metrics(&predictions, labelled_pairs, |row| row.baseline),
        candidate: metrics(&predictions, labelled_pairs, |row| row.predicted),
        common_labelled_pairs: common.len(),
        common_baseline_accuracy: fraction(common_baseline, common.len()),
        common_candidate_accuracy: fraction(common_candidate, common.len()),
        improvements: common
            .iter()
            .filter(|row| row.baseline != row.gold && row.predicted == row.gold)
            .count(),
        regressions: common
            .iter()
            .filter(|row| row.baseline == row.gold && row.predicted != row.gold)
            .count(),
        predictions,
    })
}

fn metrics(
    rows: &[ContextPairPrediction],
    labelled: usize,
    prediction: impl Fn(&ContextPairPrediction) -> Option<bool>,
) -> ContextMetrics {
    let available = rows.iter().filter(|row| prediction(row).is_some()).count();
    let labelled_available = rows
        .iter()
        .filter(|row| row.gold.is_some() && prediction(row).is_some())
        .count();
    let correct = rows
        .iter()
        .filter(|row| row.gold.is_some() && prediction(row) == row.gold)
        .count();
    ContextMetrics {
        available_pairs: available,
        coverage: fraction(available, rows.len()).unwrap_or(0.0),
        labelled_available,
        correct,
        accuracy: fraction(correct, labelled_available),
        accuracy_with_unavailable_as_errors: fraction(correct, labelled),
    }
}

fn fraction(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}
