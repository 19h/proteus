//! Corpus-adapted information content with a uniform lexical-sense prior.

use super::{invalid, SemanticModel};
use crate::Result;
use std::sync::Arc;

pub(super) fn canonical_pos(pos: &str) -> &str {
    if pos == "s" {
        "a"
    } else {
        pos
    }
}

impl SemanticModel {
    fn information_content(&self) -> Result<&[f64]> {
        self.information_content
            .get_or_init(|| {
                self.build_information_content()
                    .map(Arc::new)
                    .map_err(|error| error.to_string())
            })
            .as_ref()
            .map(|values| values.as_slice())
            .map_err(|message| invalid(message))
    }

    fn build_information_content(&self) -> Result<Vec<f64>> {
        let index = self.index()?;
        let mut masses = vec![0.0; self.lexicon.senses.len()];
        let mut totals = std::collections::BTreeMap::<&str, f64>::new();
        // Each lexical sense receives one pseudo-occurrence. Propagating it to
        // each distinct ancestor avoids zero counts and preserves monotonicity
        // under multiple inheritance without counting duplicate DAG paths.
        for (sense_id, sense) in self.lexicon.senses.iter().enumerate() {
            let pos = canonical_pos(&sense.pos);
            *totals.entry(pos).or_default() += 1.0;
            for &ancestor in index.ancestors[sense_id].keys() {
                if canonical_pos(&self.lexicon.senses[ancestor].pos) != pos {
                    return Err(invalid(
                        "Information content requires POS-consistent taxonomy edges",
                    ));
                }
                masses[ancestor] += 1.0;
            }
        }
        for entry in self.entries.values() {
            for evidence in &entry.senses {
                let sense_id = *index
                    .ids
                    .get(&evidence.sense)
                    .ok_or_else(|| invalid("Unknown sense in information-content evidence"))?;
                let weight = entry.count as f64 * evidence.prior;
                if !weight.is_finite() || weight < 0.0 {
                    return Err(invalid("Invalid information-content occurrence mass"));
                }
                let pos = canonical_pos(&self.lexicon.senses[sense_id].pos);
                *totals.entry(pos).or_default() += weight;
                for &ancestor in index.ancestors[sense_id].keys() {
                    masses[ancestor] += weight;
                }
            }
        }
        masses
            .iter()
            .enumerate()
            .map(|(sense_id, &mass)| {
                let total = totals[canonical_pos(&self.lexicon.senses[sense_id].pos)];
                if !total.is_finite()
                    || !mass.is_finite()
                    || mass <= 0.0
                    || mass > total * (1.0 + 1e-12)
                {
                    return Err(invalid("Invalid information-content probability mass"));
                }
                Ok(-(mass / total).min(1.0).ln())
            })
            .collect()
    }

    pub(super) fn information_similarity(&self, left: usize, right: usize) -> Result<f64> {
        let index = self.index()?;
        if left == right {
            return Ok(1.0);
        }
        if index.antonyms[left].contains(&right) {
            return Ok(0.0);
        }
        if index.similar[left].contains(&right) {
            return Ok(0.9);
        }
        if canonical_pos(&self.lexicon.senses[left].pos)
            != canonical_pos(&self.lexicon.senses[right].pos)
        {
            return Ok(0.0);
        }
        let information = self.information_content()?;
        let denominator = information[left] + information[right];
        if denominator == 0.0 {
            return Ok(0.0);
        }
        let shared = index.ancestors[left]
            .keys()
            .filter(|ancestor| index.ancestors[right].contains_key(ancestor))
            .map(|&ancestor| information[ancestor])
            .fold(0.0, f64::max);
        Ok((2.0 * shared / denominator).clamp(0.0, 1.0))
    }
}
