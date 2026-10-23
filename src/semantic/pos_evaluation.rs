//! Pair evaluation that preserves author-supplied grammatical annotations.

use super::{PartOfSpeech, Relation, SemanticModel};
use crate::evaluation::{evaluate_with, read_word_pairs, EvaluationReport, WordPair};
use crate::{ProteusError, Result};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Cursor};

/// One labelled pair with an explicitly supplied POS shared by both words.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PosWordPair {
    /// Original words and reference score, without normalization or deduplication.
    #[serde(flatten)]
    pub pair: WordPair,
    /// Author-supplied annotation; no tagger runs during evaluation.
    pub part_of_speech: PartOfSpeech,
}

/// Every input observation and its POS-conditioned prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PosPairPrediction {
    /// Original labelled pair and grammatical annotation.
    #[serde(flatten)]
    pub observation: PosWordPair,
    /// Missing compatible evidence remains null rather than a score of zero.
    pub predicted: Option<f64>,
}

/// Read annotated TSV pairs. Both column numbers are one-based and distinct;
/// POS accepts the normal lexical codes/names, case-insensitively. Malformed or
/// unknown annotations abort with their original line number. Blank lines and
/// explicit headers follow read_word_pairs; repeated observations are retained.
pub fn read_pos_word_pairs(
    reader: impl BufRead,
    score_column: usize,
    pos_column: usize,
    header: bool,
) -> Result<Vec<PosWordPair>> {
    if score_column < 3 || pos_column < 3 || score_column == pos_column {
        return Err(ProteusError::Config(
            "Score and POS columns must be distinct and at least 3".into(),
        ));
    }
    let mut pairs = Vec::new();
    let mut first_nonblank_line = true;
    for (line_number, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if first_nonblank_line && header {
            first_nonblank_line = false;
            continue;
        }
        first_nonblank_line = false;
        let error = |message: &str| {
            ProteusError::Config(format!(
                "Invalid POS-labelled word pair on line {}: {message}",
                line_number + 1
            ))
        };
        let annotation = line
            .split('\t')
            .nth(pos_column - 1)
            .ok_or_else(|| error("missing POS column"))?;
        let part_of_speech = annotation
            .trim()
            .to_ascii_lowercase()
            .parse()
            .map_err(|_| error("unknown POS annotation"))?;
        let pair = read_word_pairs(Cursor::new(line.as_bytes()), score_column, false)
            .map_err(|cause| error(&cause.to_string()))?
            .pop()
            .ok_or_else(|| error("missing observation"))?;
        pairs.push(PosWordPair {
            pair,
            part_of_speech,
        });
    }
    Ok(pairs)
}

/// Evaluate the same declared relation within each observation's supplied POS.
/// Every row, including duplicate words with different POS and unavailable
/// lexical evidence, remains in the report. Association is rejected even for
/// an empty input because its coordinates have no sense-level POS contract.
pub fn evaluate_pos_pairs(
    model: &SemanticModel,
    observations: &[PosWordPair],
    relation: Relation,
) -> Result<EvaluationReport<PosPairPrediction>> {
    if relation == Relation::Association {
        return Err(ProteusError::Config(
            "POS-labelled evaluation requires a lexical relation".into(),
        ));
    }
    let pairs: Vec<_> = observations.iter().map(|row| row.pair.clone()).collect();
    let mut annotations = observations.iter();
    let report = evaluate_with(&pairs, |left, right| {
        let annotation = annotations.next().ok_or_else(|| {
            ProteusError::Config("Evaluation observation ordering mismatch".into())
        })?;
        model.similarity_with_pos(left, right, relation, annotation.part_of_speech)
    })?;
    Ok(EvaluationReport {
        total_pairs: report.total_pairs,
        evaluated_pairs: report.evaluated_pairs,
        coverage: report.coverage,
        zero_similarity_pairs: report.zero_similarity_pairs,
        spearman: report.spearman,
        predictions: report
            .predictions
            .into_iter()
            .zip(observations)
            .map(|(prediction, observation)| PosPairPrediction {
                observation: observation.clone(),
                predicted: prediction.predicted,
            })
            .collect(),
    })
}
