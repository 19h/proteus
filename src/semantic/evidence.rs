//! Bound predicate/argument evidence with explicit support, conflict and unknown.

use super::{invalid, AnnotatedDocument, SemanticModel};
use crate::{Result, TextConfig};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use unicode_normalization::UnicodeNormalization;

/// Canonical role names for the supported dependency extraction patterns.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventRole {
    /// Voice-normalized active-subject slot, not a general thematic-agent label.
    ActiveSubject,
    /// Voice-normalized active-object slot, including a passive subject.
    ActiveObject,
    /// Indirect object or recipient.
    IndirectObject,
    /// Attribute or complement attached to the predicate.
    Complement,
    /// Prepositional role retaining the original preposition lemma.
    Prepositional(String),
}

/// One bound event participant with original source offsets.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Argument {
    /// Canonical participant role, independent of active versus passive voice.
    pub role: EventRole,
    /// Bound participant head or compound lemma, not a bag of all sentence nouns.
    pub lemma: String,
    /// Inclusive UTF-8 source byte offset.
    pub start: usize,
    /// Exclusive UTF-8 source byte offset.
    pub end: usize,
    /// Explicit determiners and numeric modifiers; no quantifier inference is assumed.
    pub quantifiers: Vec<String>,
}

/// A source assertion with bound participants, polarity and scope qualifiers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Event {
    /// Identity of the original source document.
    pub document: String,
    /// Parser sentence identity within the document.
    pub sentence: usize,
    /// Inclusive UTF-8 source byte offset.
    pub start: usize,
    /// Exclusive UTF-8 source byte offset.
    pub end: usize,
    /// Parser lemma of this verbal predicate.
    pub predicate: String,
    /// Participants bound to this event, each with its own role and source span.
    pub arguments: Vec<Argument>,
    /// Explicit predicate-level negation in the supported extraction patterns.
    pub negated: bool,
    /// Retained tense, modal and adverb qualifiers required for evidence matching.
    pub qualifiers: Vec<String>,
    /// False when scope or participant count exceeds the supported extraction patterns.
    pub scope_complete: bool,
}

impl Event {
    pub(super) fn validate(&self, documents: &BTreeMap<&str, &AnnotatedDocument>) -> Result<()> {
        let document = documents
            .get(self.document.as_str())
            .ok_or_else(|| invalid("Event references unknown document"))?;
        if self.predicate.is_empty()
            || self.start >= self.end
            || document.text.get(self.start..self.end).is_none()
            || self.arguments.iter().any(|argument| {
                argument.lemma.is_empty()
                    || argument.start < self.start
                    || argument.end > self.end
                    || argument.start >= argument.end
                    || document.text.get(argument.start..argument.end).is_none()
            })
        {
            return Err(invalid("Invalid event evidence or argument range"));
        }
        Ok(())
    }
}

/// Evidence state for a required query assertion.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStatus {
    /// Matching bound assertion with the requested polarity and qualifiers.
    Supported,
    /// Matching bound assertion with explicit opposite polarity.
    Conflicted,
    /// No matching supported or conflicting assertion was extracted.
    Unknown,
}

/// Directional query coverage and inspectable source evidence for one document.
#[derive(Debug, Clone, Serialize)]
pub struct EvidenceMatch {
    /// Identity of the original source document.
    pub document: String,
    /// Original unmodified source text; byte ranges refer to this string.
    pub text: String,
    /// Directional support-minus-conflict score; not a truth probability.
    pub score: f64,
    /// Number of query events supported by matching bound assertions.
    pub supported: usize,
    /// Number of query events with an explicit matching polarity conflict.
    pub conflicted: usize,
    /// Number of query events without matching support or conflict.
    pub unknown: usize,
    /// One status per query event. Evidence locations remain source-attributed.
    pub evidence: Vec<(EvidenceStatus, Option<Event>)>,
}

pub(super) fn extract_events(document: &AnnotatedDocument, text_config: &TextConfig) -> Vec<Event> {
    let tokens = &document.tokens;
    let mut children = vec![Vec::new(); tokens.len()];
    let mut sentence_ranges = BTreeMap::<usize, (usize, usize)>::new();
    let mut negation_heads = BTreeMap::<usize, Vec<usize>>::new();
    for (position, token) in tokens.iter().enumerate() {
        if matches!(token.lemma.as_str(), "not" | "never" | "neither" | "no") {
            negation_heads
                .entry(token.sentence)
                .or_default()
                .push(token.head);
        }
        sentence_ranges
            .entry(token.sentence)
            .and_modify(|range| range.1 = token.end)
            .or_insert((token.start, token.end));
        if token.head != position {
            children[token.head].push(position);
        }
    }
    let mut events = Vec::new();
    for (position, token) in tokens.iter().enumerate() {
        if token.pos != "VERB" {
            continue;
        }
        let &(sentence_start, sentence_end) = &sentence_ranges[&token.sentence];
        let passive = children[position].iter().any(|&child| {
            matches!(
                tokens[child].dependency.as_str(),
                "nsubjpass" | "nsubj:pass" | "auxpass" | "aux:pass"
            )
        });
        let mut arguments = Vec::new();
        let mut negated = false;
        let mut qualifiers = Vec::new();
        let mut scope_complete = negation_heads
            .get(&token.sentence)
            .is_none_or(|heads| heads.iter().all(|&head| head == position));
        let finite = children[position]
            .iter()
            .copied()
            .filter(|&child| {
                matches!(
                    tokens[child].dependency.as_str(),
                    "aux" | "auxpass" | "aux:pass"
                )
            })
            .find(|&child| {
                tokens[child]
                    .morphology
                    .iter()
                    .any(|feature| feature == "VerbForm=Fin")
            })
            .unwrap_or(position);
        qualifiers.extend(
            tokens[finite]
                .morphology
                .iter()
                .filter(|feature| feature.starts_with("Tense="))
                .cloned(),
        );
        for &child in &children[position] {
            let dependent = &tokens[child];
            if dependent.dependency == "neg" || matches!(dependent.lemma.as_str(), "not" | "never")
            {
                negated = true;
                continue;
            }
            if matches!(
                dependent.lemma.as_str(),
                "can" | "could" | "may" | "might" | "must" | "should" | "would" | "will"
            ) {
                qualifiers.push(format!("modal:{}", dependent.lemma));
            } else if dependent.pos == "ADV" {
                qualifiers.push(format!("adverb:{}", dependent.lemma));
            }
            let role = match dependent.dependency.as_str() {
                "nsubj" | "csubj" => Some(EventRole::ActiveSubject),
                "nsubjpass" | "nsubj:pass" | "obj" | "dobj" => Some(EventRole::ActiveObject),
                "iobj" => Some(EventRole::IndirectObject),
                "attr" | "acomp" => Some(EventRole::Complement),
                "obl:agent" => Some(EventRole::ActiveSubject),
                _ => None,
            };
            if let Some(role) = role {
                arguments.push(argument(document, child, role, &children, text_config));
            } else if matches!(dependent.dependency.as_str(), "prep" | "agent") {
                for &object in &children[child] {
                    if matches!(tokens[object].dependency.as_str(), "pobj" | "obj") {
                        let role = if passive && dependent.lemma == "by" {
                            EventRole::ActiveSubject
                        } else {
                            EventRole::Prepositional(dependent.lemma.to_lowercase())
                        };
                        arguments.push(argument(document, object, role, &children, text_config));
                    }
                }
            }
        }
        arguments.sort();
        arguments.dedup();
        if arguments.len() > 64 {
            scope_complete = false;
        }
        qualifiers.sort();
        qualifiers.dedup();
        if !arguments.is_empty() {
            events.push(Event {
                document: document.id.clone(),
                sentence: token.sentence,
                start: sentence_start,
                end: sentence_end,
                predicate: normalize_binding(&token.lemma, text_config),
                arguments,
                negated,
                qualifiers,
                scope_complete,
            });
        }
    }
    events
}

fn argument(
    document: &AnnotatedDocument,
    head: usize,
    role: EventRole,
    children: &[Vec<usize>],
    text_config: &TextConfig,
) -> Argument {
    let mut components = vec![head];
    let mut quantifiers = Vec::new();
    for &child in &children[head] {
        let dependent = &document.tokens[child];
        if matches!(dependent.dependency.as_str(), "compound" | "flat" | "name") {
            components.push(child);
        } else if matches!(dependent.dependency.as_str(), "det" | "nummod" | "quantmod") {
            quantifiers.push(dependent.lemma.to_lowercase());
        }
    }
    components.sort_unstable();
    quantifiers.sort();
    Argument {
        role,
        lemma: components
            .iter()
            .map(|&position| normalize_binding(&document.tokens[position].lemma, text_config))
            .collect::<Vec<_>>()
            .join(" "),
        start: document.tokens[*components.first().unwrap_or(&head)].start,
        end: document.tokens[*components.last().unwrap_or(&head)].end,
        quantifiers,
    }
}

// Binding identities retain one-letter names, punctuation and numbers even when
// lexical vocabulary filters exclude them. Case and Unicode policy remain shared.
fn normalize_binding(value: &str, config: &TextConfig) -> String {
    let normalized = if config.unicode_normalize {
        value.nfd().collect::<String>()
    } else {
        value.to_owned()
    };
    if config.lowercase {
        normalized.to_lowercase()
    } else {
        normalized
    }
}

impl SemanticModel {
    /// Search attributed events. Reversed roles are unknown support; an explicit
    /// polarity reversal of the same bound assertion is conflicting evidence.
    pub fn search_evidence(
        &self,
        query: &AnnotatedDocument,
        k: usize,
        conflict_penalty: f64,
    ) -> Result<Vec<EvidenceMatch>> {
        query.validate()?;
        let index = self.index()?;
        if !conflict_penalty.is_finite() || conflict_penalty < 0.0 {
            return Err(crate::ProteusError::Config(
                "Conflict penalty must be finite and nonnegative".into(),
            ));
        }
        let requirements = extract_events(query, &self.report.config.text);
        if requirements.is_empty() {
            return Err(crate::ProteusError::EmptyInput(
                "Query contains no supported event structure".into(),
            ));
        }
        if requirements.iter().any(|event| !event.scope_complete) {
            return Err(crate::ProteusError::Config(
                "Query scope or participant count is unsupported by this extractor".into(),
            ));
        }
        let source_documents: BTreeMap<_, _> = self
            .documents
            .iter()
            .map(|document| (document.id.as_str(), document))
            .collect();
        let mut by_document: BTreeMap<&str, Vec<&Event>> = self
            .documents
            .iter()
            .map(|document| (document.id.as_str(), Vec::new()))
            .collect();
        for event in &self.events {
            by_document
                .get_mut(event.document.as_str())
                .ok_or_else(|| invalid("Event references an unknown source document"))?
                .push(event);
        }
        let mut results = Vec::new();
        for (document, events) in by_document {
            let mut evidence = Vec::new();
            for required in &requirements {
                let matching: Vec<_> = events
                    .iter()
                    .copied()
                    .filter(|candidate| self.same_assertion(required, candidate, index))
                    .collect();
                let positive = matching
                    .iter()
                    .find(|event| event.negated == required.negated);
                let negative = matching
                    .iter()
                    .find(|event| event.negated != required.negated);
                // Conflicting assertions within a document remain a conflict,
                // rather than disappearing behind one supporting occurrence.
                evidence.push(if let Some(event) = negative {
                    (EvidenceStatus::Conflicted, Some((**event).clone()))
                } else if let Some(event) = positive {
                    (EvidenceStatus::Supported, Some((**event).clone()))
                } else {
                    (EvidenceStatus::Unknown, None)
                });
            }
            let supported = evidence
                .iter()
                .filter(|(status, _)| *status == EvidenceStatus::Supported)
                .count();
            let conflicted = evidence
                .iter()
                .filter(|(status, _)| *status == EvidenceStatus::Conflicted)
                .count();
            results.push(EvidenceMatch {
                document: document.into(),
                text: source_documents[document].text.clone(),
                score: (supported as f64 - conflict_penalty * conflicted as f64)
                    / requirements.len() as f64,
                supported,
                conflicted,
                unknown: requirements.len() - supported - conflicted,
                evidence,
            });
        }
        results.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then(left.document.cmp(&right.document))
        });
        results.truncate(k);
        Ok(results)
    }

    fn same_assertion(
        &self,
        query: &Event,
        candidate: &Event,
        index: &super::lexicon::LexiconIndex,
    ) -> bool {
        let synonymous = |left: &str, right: &str| {
            left == right
                || index
                    .lemmas
                    .get(left)
                    .zip(index.lemmas.get(right))
                    .is_some_and(|(left, right)| left.iter().any(|sense| right.contains(sense)))
        };
        query.scope_complete
            && candidate.scope_complete
            && query.qualifiers == candidate.qualifiers
            && synonymous(&query.predicate, &candidate.predicate)
            && query.arguments.len() == candidate.arguments.len()
            && match_arguments(&query.arguments, &candidate.arguments, synonymous)
    }
}

fn match_arguments(
    query: &[Argument],
    candidate: &[Argument],
    synonymous: impl Fn(&str, &str) -> bool,
) -> bool {
    if query.len() != candidate.len() || query.len() > 64 {
        return false;
    }
    let edges: Vec<Vec<usize>> = query
        .iter()
        .map(|argument| {
            candidate
                .iter()
                .enumerate()
                .filter_map(|(position, other)| {
                    (argument.role == other.role
                        && argument.quantifiers == other.quantifiers
                        && synonymous(&argument.lemma, &other.lemma))
                    .then_some(position)
                })
                .collect()
        })
        .collect();
    let mut owners = vec![None; candidate.len()];
    for position in 0..query.len() {
        let mut visited = vec![false; candidate.len()];
        if !augment_matching(position, &edges, &mut visited, &mut owners) {
            return false;
        }
    }
    true
}

// The supported participant bound limits recursion to 64 levels. Augmenting
// paths provide a bijection even when several lexical alternatives overlap.
fn augment_matching(
    query: usize,
    edges: &[Vec<usize>],
    visited: &mut [bool],
    owners: &mut [Option<usize>],
) -> bool {
    for &candidate in &edges[query] {
        if visited[candidate] {
            continue;
        }
        visited[candidate] = true;
        if owners[candidate]
            .is_none_or(|previous| augment_matching(previous, edges, visited, owners))
        {
            owners[candidate] = Some(query);
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{match_arguments, Argument, EventRole};

    fn participant(lemma: &str) -> Argument {
        Argument {
            role: EventRole::ActiveObject,
            lemma: lemma.into(),
            start: 0,
            end: lemma.len(),
            quantifiers: Vec::new(),
        }
    }

    #[test]
    fn participant_assignment_is_injective_and_revisits_ambiguous_matches() {
        let candidate = [participant("alpha"), participant("beta")];
        let repeated = [participant("alpha"), participant("alpha")];
        assert!(!match_arguments(&repeated, &candidate, |left, right| {
            left == right
        }));

        // The first flexible match must move from alpha to beta to leave the
        // only valid participant available for the second requirement.
        let ambiguous = [participant("flexible"), participant("alpha")];
        assert!(match_arguments(&ambiguous, &candidate, |left, right| {
            left == "flexible" || left == right
        }));
    }
}
