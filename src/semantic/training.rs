//! Deterministic corpus adaptation of explicit lexical senses and event evidence.

use super::corpus::read_annotated_corpus;
use super::evidence::extract_events;
use super::{
    Lexicon, OccurrenceReference, SemanticConfig, SemanticEntry, SemanticModel, SemanticReport,
    SenseEvidence, SenseInventory,
};
use crate::{Result, Retina, Tokenizer, WordFingerprint};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::BufRead;
use std::sync::{Arc, OnceLock};

struct Occurrence {
    word: String,
    lemma: String,
    token: usize,
    last_token: usize,
    start: usize,
    end: usize,
}

#[derive(Default)]
struct WordCounts {
    count: usize,
    sense_occurrences: usize,
    lemmas: BTreeSet<String>,
    senses: BTreeSet<usize>,
}

#[derive(Default)]
struct SenseCounts {
    mass: f64,
    documents: BTreeSet<String>,
    examples: Vec<(f64, OccurrenceReference)>,
}

/// Train from complete, attributed dependency annotations and a validated lexical
/// source. The returned v5 retina preserves graph semantics and source evidence.
pub fn train_semantic(
    reader: impl BufRead,
    lexicon: Lexicon,
    config: SemanticConfig,
) -> Result<(Retina, SemanticReport)> {
    config.validate()?;
    let documents = read_annotated_corpus(reader)?;
    let index = lexicon.build_index(&config.text)?;
    let tokenizer = Tokenizer::new(config.text.clone());
    let mut counts = BTreeMap::<String, WordCounts>::new();
    let mut corpus_digest = Sha256::new();
    let mut tokens = 0;
    for document in &documents {
        let serialized =
            serde_json::to_vec(document).map_err(|error| super::invalid(&error.to_string()))?;
        corpus_digest.update((serialized.len() as u64).to_le_bytes());
        corpus_digest.update(serialized);
        tokens += document
            .tokens
            .iter()
            .filter(|token| {
                !tokenizer
                    .tokenize_to_strings(&document.text[token.start..token.end])
                    .is_empty()
            })
            .count();
        for occurrence in occurrences(document, &tokenizer, &index, config.max_phrase_tokens) {
            let row = counts.entry(occurrence.word.clone()).or_default();
            row.count += 1;
            row.lemmas.insert(occurrence.lemma.clone());
            match config.sense_inventory {
                SenseInventory::Lexical => {
                    for key in [&occurrence.word, &occurrence.lemma] {
                        if let Some(senses) = index.lemmas.get(key) {
                            row.senses.extend(senses);
                        }
                    }
                }
                SenseInventory::Observed => {
                    let candidates = observed_senses(&occurrence, document, &index, &lexicon);
                    if !candidates.is_empty() {
                        row.sense_occurrences += 1;
                        row.senses.extend(candidates);
                    }
                }
            }
        }
    }
    counts.retain(|_, row| row.count >= config.min_count);
    if counts.is_empty() {
        return Err(crate::ProteusError::EmptyInput(
            "No semantic vocabulary meets min_count".into(),
        ));
    }
    let mut source_priors = BTreeMap::new();
    let mut evidence = BTreeMap::<String, BTreeMap<usize, SenseCounts>>::new();
    for (word, row) in &counts {
        let mut candidates: Vec<_> = row
            .senses
            .iter()
            .map(|&sense| {
                let lexical_count: f64 = lexicon.senses[sense]
                    .lemmas
                    .iter()
                    .filter(|lemma| {
                        let key = tokenizer.tokenize_to_strings(&lemma.text).join(" ");
                        key == *word || row.lemmas.contains(&key)
                    })
                    .map(|lemma| lemma.count as f64)
                    .sum();
                (sense, lexical_count + 1.0)
            })
            .collect();
        let total: f64 = candidates.iter().map(|(_, mass)| mass).sum();
        for (_, mass) in &mut candidates {
            *mass /= total;
        }
        evidence.insert(
            word.clone(),
            candidates
                .iter()
                .map(|(sense, _)| (*sense, SenseCounts::default()))
                .collect(),
        );
        source_priors.insert(word.clone(), candidates);
    }
    let mut cooccurrences = BTreeMap::<String, BTreeMap<String, f64>>::new();
    let mut context_totals = BTreeMap::<String, f64>::new();
    let mut events = Vec::new();
    for document in &documents {
        let source_group = format!("{:x}", Sha256::digest(document.text.as_bytes()));
        let normalized: Vec<_> = document
            .tokens
            .iter()
            .map(|token| {
                tokenizer
                    .tokenize_to_strings(&document.text[token.start..token.end])
                    .join(" ")
            })
            .collect();
        let lemmas: Vec<_> = document
            .tokens
            .iter()
            .map(|token| tokenizer.tokenize_to_strings(&token.lemma).join(" "))
            .collect();
        for occurrence in occurrences(document, &tokenizer, &index, config.max_phrase_tokens) {
            let Some(candidates) = source_priors.get(&occurrence.word) else {
                continue;
            };
            let start = occurrence.token.saturating_sub(config.context_window);
            let end = occurrence
                .last_token
                .saturating_add(config.context_window)
                .saturating_add(1)
                .min(document.tokens.len());
            let mut context = Vec::new();
            for position in start..end {
                if (occurrence.token..=occurrence.last_token).contains(&position)
                    || document.tokens[position].sentence
                        != document.tokens[occurrence.token].sentence
                {
                    continue;
                }
                context.push(lemmas[position].as_str());
                let context_word = &normalized[position];
                if !counts.contains_key(context_word) {
                    continue;
                }
                let distance = if position < occurrence.token {
                    occurrence.token - position
                } else {
                    position - occurrence.last_token
                };
                let weight =
                    (config.context_window + 1 - distance) as f64 / config.context_window as f64;
                *cooccurrences
                    .entry(occurrence.word.clone())
                    .or_default()
                    .entry(context_word.clone())
                    .or_default() += weight;
                *context_totals.entry(context_word.clone()).or_default() += weight;
            }
            let conditioned;
            let candidates = if config.sense_inventory == SenseInventory::Observed {
                let allowed = observed_senses(&occurrence, document, &index, &lexicon);
                conditioned = candidates
                    .iter()
                    .copied()
                    .filter(|(sense, _)| allowed.contains(sense))
                    .collect::<Vec<_>>();
                conditioned.as_slice()
            } else {
                candidates.as_slice()
            };
            if candidates.is_empty() {
                continue;
            }
            let posterior = index.posterior(candidates, &context.join(" "), &config);
            for (&(sense, _), probability) in candidates.iter().zip(posterior) {
                let statistics = evidence
                    .get_mut(&occurrence.word)
                    .and_then(|row| row.get_mut(&sense))
                    .ok_or_else(|| super::invalid("Missing sense accumulator"))?;
                statistics.mass += probability;
                if probability >= config.min_sense_support {
                    statistics.documents.insert(source_group.clone());
                    statistics.examples.push((
                        probability,
                        OccurrenceReference {
                            document: document.id.clone(),
                            start: occurrence.start,
                            end: occurrence.end,
                        },
                    ));
                    statistics.examples.sort_by(|left, right| {
                        right
                            .0
                            .total_cmp(&left.0)
                            .then(left.1.document.cmp(&right.1.document))
                            .then(left.1.start.cmp(&right.1.start))
                    });
                    statistics.examples.truncate(config.max_examples_per_sense);
                }
            }
        }
        events.extend(extract_events(document, &config.text));
    }
    let normalizer: f64 = context_totals
        .values()
        .map(|mass| mass.powf(config.context_alpha))
        .sum();
    let mut entries = BTreeMap::new();
    for (word, row) in counts {
        let candidates = &source_priors[&word];
        let mut senses = Vec::new();
        let sense_count = match config.sense_inventory {
            SenseInventory::Lexical => row.count,
            SenseInventory::Observed => row.sense_occurrences,
        };
        for &(sense, prior) in candidates {
            let statistics = &evidence[&word][&sense];
            senses.push(SenseEvidence {
                sense: lexicon.senses[sense].id.clone(),
                prior: (config.lexical_prior_strength * prior + statistics.mass)
                    / (config.lexical_prior_strength + sense_count as f64),
                occurrence_mass: statistics.mass,
                supporting_documents: statistics.documents.len(),
                examples: statistics
                    .examples
                    .iter()
                    .map(|(_, example)| example.clone())
                    .collect(),
            });
        }
        let mut associations = Vec::new();
        if let Some(contexts) = cooccurrences.get(&word) {
            let total: f64 = contexts.values().sum();
            for (context, &mass) in contexts {
                let background = context_totals[context].powf(config.context_alpha) / normalizer;
                let score = ((mass / total) / background).ln().max(0.0);
                if score > 0.0 {
                    associations.push((context.clone(), score));
                }
            }
        }
        associations.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
        associations.truncate(config.max_associations);
        let norm = associations
            .iter()
            .map(|(_, value)| value * value)
            .sum::<f64>()
            .sqrt();
        let associations = associations
            .into_iter()
            .map(|(feature, value)| (feature, value / norm))
            .collect();
        entries.insert(
            word,
            SemanticEntry {
                count: row.count,
                lemmas: row.lemmas.into_iter().collect(),
                senses,
                sense_occurrences: (config.sense_inventory == SenseInventory::Observed)
                    .then_some(row.sense_occurrences),
                associations,
            },
        );
    }
    let lexicon_bytes =
        serde_json::to_vec(&lexicon).map_err(|error| super::invalid(&error.to_string()))?;
    let report = SemanticReport {
        algorithm: config.sense_inventory.algorithm().into(),
        config: config.clone(),
        corpus_sha256: format!("{:x}", corpus_digest.finalize()),
        lexicon_sha256: format!("{:x}", Sha256::digest(lexicon_bytes)),
        documents: documents.len(),
        tokens,
        vocabulary: entries.len(),
        words_with_senses: entries
            .values()
            .filter(|entry| !entry.senses.is_empty())
            .count(),
        events: events.len(),
        parsers: documents
            .iter()
            .map(|document| document.parser.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
    };
    let cells = config.dimension * config.dimension;
    let cap = config.fingerprint.active_bit_limit(cells as u32);
    let total_mass: usize = entries.values().map(|entry| entry.count).sum();
    let mut fingerprints = HashMap::new();
    for (word, entry) in &entries {
        let mut features = BTreeMap::<u32, f64>::new();
        for sense in &entry.senses {
            for (&ancestor, &distance) in &index.ancestors[index.ids[&sense.sense]] {
                let digest = Sha256::digest(lexicon.senses[ancestor].id.as_bytes());
                let slot = (u64::from_le_bytes(
                    digest[..8]
                        .try_into()
                        .map_err(|_| super::invalid("Invalid feature digest"))?,
                ) % cells as u64) as u32;
                *features.entry(slot).or_default() +=
                    sense.prior * index.depth[ancestor] as f64 / (distance as f64 + 1.0);
            }
        }
        let mut ranked: Vec<_> = features.into_iter().collect();
        ranked.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
        let positions: Vec<_> = ranked
            .into_iter()
            .take(cap)
            .map(|(position, _)| position)
            .collect();
        fingerprints.insert(
            word.clone(),
            WordFingerprint::with_frequency(
                word.clone(),
                &positions,
                cells as u32,
                entry.count as f64 / total_mass as f64,
            ),
        );
    }
    let model = SemanticModel {
        report: report.clone(),
        lexicon,
        entries,
        documents,
        events,
        index: OnceLock::from(Ok(Arc::new(index))),
        information_content: OnceLock::new(),
    };
    model.validate()?;
    let mut retina = Retina::with_index(fingerprints, config.dimension as u32, config.fingerprint);
    retina.set_semantic_model(model)?;
    Ok((retina, report))
}

fn occurrences(
    document: &super::AnnotatedDocument,
    tokenizer: &Tokenizer,
    index: &super::lexicon::LexiconIndex,
    max_phrase_tokens: usize,
) -> Vec<Occurrence> {
    let mut output = Vec::new();
    for (position, token) in document.tokens.iter().enumerate() {
        let word = tokenizer
            .tokenize_to_strings(&document.text[token.start..token.end])
            .join(" ");
        if word.is_empty() {
            continue;
        }
        let lemma = tokenizer.tokenize_to_strings(&token.lemma).join(" ");
        output.push(Occurrence {
            word,
            lemma,
            token: position,
            last_token: position,
            start: token.start,
            end: token.end,
        });
        for last in position + 1
            ..position
                .saturating_add(max_phrase_tokens)
                .min(document.tokens.len())
        {
            let final_token = &document.tokens[last];
            if final_token.sentence != token.sentence {
                break;
            }
            if final_token.pos == "PUNCT" || final_token.pos == "SPACE" {
                continue;
            }
            let phrase = tokenizer
                .tokenize_to_strings(&document.text[token.start..final_token.end])
                .join(" ");
            if phrase.contains(' ') && index.lemmas.contains_key(&phrase) {
                output.push(Occurrence {
                    word: phrase.clone(),
                    lemma: phrase,
                    token: position,
                    last_token: last,
                    start: token.start,
                    end: final_token.end,
                });
            }
        }
    }
    output
}

/// A phrase has an attributable POS only when its span has one dependency head.
/// Self-headed roots count as external heads. Coordination and disconnected
/// fragments with several heads remain unavailable rather than guessed.
fn occurrence_head<'a>(
    occurrence: &Occurrence,
    document: &'a super::AnnotatedDocument,
) -> Option<&'a super::AnnotatedToken> {
    let mut heads = (occurrence.token..=occurrence.last_token).filter(|&position| {
        let head = document.tokens[position].head;
        head == position || !(occurrence.token..=occurrence.last_token).contains(&head)
    });
    let head = heads.next()?;
    heads.next().is_none().then_some(&document.tokens[head])
}

fn observed_senses(
    occurrence: &Occurrence,
    document: &super::AnnotatedDocument,
    index: &super::lexicon::LexiconIndex,
    lexicon: &Lexicon,
) -> Vec<usize> {
    let Some(head) = occurrence_head(occurrence, document) else {
        return Vec::new();
    };
    let pos = match head.pos.as_str() {
        "NOUN" | "PROPN" => "n",
        "VERB" | "AUX" => "v",
        "ADJ" => "a",
        "ADV" => "r",
        _ => return Vec::new(),
    };
    // An available parser lemma takes precedence over an ambiguous surface
    // spelling: saw/see must not acquire the unrelated saw/cutting sense.
    index
        .lemmas
        .get(&occurrence.lemma)
        .or_else(|| index.lemmas.get(&occurrence.word))
        .into_iter()
        .flatten()
        .copied()
        .filter(|&sense| super::information::canonical_pos(&lexicon.senses[sense].pos) == pos)
        .collect()
}

/// Reconstruct the claimed inventory from retained source annotations. This
/// validates attestation, not the linguistic accuracy of the parser's labels.
pub(super) fn validate_observed_inventory(
    model: &SemanticModel,
    index: &super::lexicon::LexiconIndex,
    tokenizer: &Tokenizer,
) -> Result<()> {
    let mut expected = BTreeMap::<String, WordCounts>::new();
    for document in &model.documents {
        for occurrence in occurrences(
            document,
            tokenizer,
            index,
            model.report.config.max_phrase_tokens,
        ) {
            let candidates = observed_senses(&occurrence, document, index, &model.lexicon);
            let row = expected.entry(occurrence.word).or_default();
            row.count += 1;
            row.lemmas.insert(occurrence.lemma);
            if !candidates.is_empty() {
                row.sense_occurrences += 1;
                row.senses.extend(candidates);
            }
        }
    }
    expected.retain(|_, row| row.count >= model.report.config.min_count);
    if expected.len() != model.entries.len() {
        return Err(super::invalid(
            "Observed vocabulary differs from source annotations",
        ));
    }
    for (word, row) in expected {
        let entry = model
            .entries
            .get(&word)
            .ok_or_else(|| super::invalid("Missing observed vocabulary entry"))?;
        let expected_senses: BTreeSet<_> = row
            .senses
            .iter()
            .map(|&sense| model.lexicon.senses[sense].id.as_str())
            .collect();
        let actual_senses: BTreeSet<_> = entry
            .senses
            .iter()
            .map(|evidence| evidence.sense.as_str())
            .collect();
        if entry.count != row.count
            || entry.sense_occurrences != Some(row.sense_occurrences)
            || entry.lemmas != row.lemmas.into_iter().collect::<Vec<_>>()
            || actual_senses != expected_senses
        {
            return Err(super::invalid(
                "Observed sense inventory differs from source annotations",
            ));
        }
    }
    Ok(())
}
