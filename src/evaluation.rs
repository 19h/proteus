//! Labelled pair evaluation with explicit missing-vocabulary coverage.

use crate::{ProteusError, Result, Retina};
use serde::{Deserialize, Serialize};
use std::io::BufRead;

/// A human-scored word pair; the score scale need only preserve rank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordPair {
    /// First word.
    pub word1: String,
    /// Second word.
    pub word2: String,
    /// Finite reference score.
    pub score: f64,
}

/// Reference and prediction, retaining missing pairs instead of silently dropping them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairPrediction {
    /// Labelled pair.
    #[serde(flatten)]
    pub pair: WordPair,
    /// Binary cosine, or null if either normalized word is absent.
    pub predicted: Option<f64>,
}

/// Rank agreement is conditional on coverage; neither is a training objective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    /// All valid input pairs.
    pub total_pairs: usize,
    /// Pairs where both words exist in the retina.
    pub evaluated_pairs: usize,
    /// Fraction of input pairs evaluated; zero for an empty set.
    pub coverage: f64,
    /// Covered pairs with exactly zero binary cosine.
    pub zero_similarity_pairs: usize,
    /// Pearson correlation of average ranks. Null for fewer than two pairs or constant ranks.
    pub spearman: Option<f64>,
    /// Every pair, in input order, for common-coverage comparisons and resampling.
    pub predictions: Vec<PairPrediction>,
}

/// Read UTF-8 tab-separated pairs. Word columns are 1 and 2, `score_column` is 1-based.
/// Headers are skipped only when explicitly requested. Blank lines are ignored.
pub fn read_word_pairs(
    reader: impl BufRead,
    score_column: usize,
    header: bool,
) -> Result<Vec<WordPair>> {
    if score_column < 3 {
        return Err(ProteusError::Config(
            "Score column must be at least 3".into(),
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
        let columns: Vec<_> = line.split('\t').map(str::trim).collect();
        let error =
            || ProteusError::Config(format!("Invalid word pair on line {}", line_number + 1));
        if columns.len() < score_column || columns[0].is_empty() || columns[1].is_empty() {
            return Err(error());
        }
        let score: f64 = columns[score_column - 1].parse().map_err(|_| error())?;
        if !score.is_finite() {
            return Err(error());
        }
        pairs.push(WordPair {
            word1: columns[0].into(),
            word2: columns[1].into(),
            score,
        });
    }
    Ok(pairs)
}

/// Score all labelled pairs with the same normalization and cosine used by queries.
/// O(P log P) time and O(P) memory for P pairs, excluding bitmap intersection cost.
pub fn evaluate_pairs(retina: &Retina, pairs: &[WordPair]) -> Result<EvaluationReport> {
    let mut predictions = Vec::with_capacity(pairs.len());
    let mut scores = Vec::new();
    let mut predicted_scores = Vec::new();
    for pair in pairs {
        if !pair.score.is_finite() {
            return Err(ProteusError::Config("Nonfinite reference score".into()));
        }
        let predicted = match retina.word_similarity(&pair.word1, &pair.word2) {
            Ok(value) => Some(value),
            Err(ProteusError::WordNotFound(_)) => None,
            Err(error) => return Err(error),
        };
        if let Some(value) = predicted {
            scores.push(pair.score);
            predicted_scores.push(value);
        }
        predictions.push(PairPrediction {
            pair: pair.clone(),
            predicted,
        });
    }
    Ok(EvaluationReport {
        total_pairs: pairs.len(),
        evaluated_pairs: scores.len(),
        coverage: if pairs.is_empty() {
            0.0
        } else {
            scores.len() as f64 / pairs.len() as f64
        },
        zero_similarity_pairs: predicted_scores
            .iter()
            .filter(|&&score| score == 0.0)
            .count(),
        spearman: spearman(&scores, &predicted_scores),
        predictions,
    })
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut sorted_indices: Vec<usize> = (0..values.len()).collect();
    sorted_indices.sort_unstable_by(|&left, &right| {
        values[left]
            .partial_cmp(&values[right])
            .expect("Evaluation scores must be finite")
            .then(left.cmp(&right))
    });

    let mut ranks = vec![0.0; values.len()];
    let mut group_start = 0;
    while group_start < sorted_indices.len() {
        let mut group_end = group_start + 1;
        while group_end < sorted_indices.len()
            && values[sorted_indices[group_end]] == values[sorted_indices[group_start]]
        {
            group_end += 1;
        }

        let average_rank = (group_start + group_end - 1) as f64 / 2.0;
        for &index in &sorted_indices[group_start..group_end] {
            ranks[index] = average_rank;
        }
        group_start = group_end;
    }
    ranks
}

fn spearman(reference_scores: &[f64], predicted_scores: &[f64]) -> Option<f64> {
    if reference_scores.len() < 2 || reference_scores.len() != predicted_scores.len() {
        return None;
    }

    let reference_ranks = average_ranks(reference_scores);
    let predicted_ranks = average_ranks(predicted_scores);
    let mean_rank = (reference_ranks.len() - 1) as f64 / 2.0;
    let mut covariance = 0.0;
    let mut reference_variance = 0.0;
    let mut predicted_variance = 0.0;

    for (reference_rank, predicted_rank) in reference_ranks.iter().zip(&predicted_ranks) {
        let reference_deviation = reference_rank - mean_rank;
        let predicted_deviation = predicted_rank - mean_rank;
        covariance += reference_deviation * predicted_deviation;
        reference_variance += reference_deviation.powi(2);
        predicted_variance += predicted_deviation.powi(2);
    }

    if reference_variance == 0.0 || predicted_variance == 0.0 {
        return None;
    }
    Some((covariance / (reference_variance * predicted_variance).sqrt()).clamp(-1.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn tied_values_receive_the_average_of_their_rank_positions() {
        let scores = [10.0, 20.0, 20.0, 30.0];
        let expected_ranks = vec![0.0, 1.5, 1.5, 3.0];

        assert_eq!(average_ranks(&scores), expected_ranks);
        assert_eq!(average_ranks(&[-0.0, 0.0]), vec![0.5, 0.5]);
    }

    #[test]
    fn spearman_matches_independent_pearson_calculation_on_tied_ranks() {
        let reference_scores = [1.0, 2.0, 2.0, 3.0];
        let predicted_scores = [1.0, 2.0, 3.0, 4.0];
        // Centered rank cross-product is 4.5; squared rank deviations sum to 4.5 and 5.
        let expected_correlation = 4.5 / (4.5_f64 * 5.0).sqrt();
        let actual_correlation = spearman(&reference_scores, &predicted_scores).unwrap();

        assert!((actual_correlation - expected_correlation).abs() < 1e-14);
        assert_eq!(spearman(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]), Some(-1.0));
    }

    #[test]
    fn constant_or_insufficient_scores_have_no_rank_correlation() {
        assert_eq!(spearman(&[1.0, 1.0], &[1.0, 2.0]), None);
        assert_eq!(spearman(&[], &[]), None);
        assert_eq!(spearman(&[1.0], &[2.0]), None);
    }

    #[test]
    fn pair_reader_selects_the_requested_score_column() {
        let input = "word1\tword2\tpos\tscore\nking\tqueen\tN\t9.5\n";
        let pairs = read_word_pairs(Cursor::new(input), 4, true).unwrap();

        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].word1, "king");
        assert_eq!(pairs[0].word2, "queen");
        assert_eq!(pairs[0].score, 9.5);
    }

    #[test]
    fn pair_reader_rejects_malformed_or_nonfinite_reference_scores() {
        for input in [
            "king\tqueen\tNaN",
            "king queen 5",
            "\tqueen\t5",
            "king\tqueen\tinf",
        ] {
            assert!(read_word_pairs(Cursor::new(input), 3, false).is_err());
        }
    }
}
