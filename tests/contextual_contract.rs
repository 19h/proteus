//! Independent sense, information-content and complete-observation contracts.

use proteus::semantic::{
    evaluate_context_pairs, read_context_pairs, train_semantic, AnnotatedDocument, AnnotatedToken,
    ByteSpan, ContextPair, ContextualWord, LexicalSense, Lexicon, LexiconSource, ParserProvenance,
    PartOfSpeech, Relation, SemanticConfig, SenseLemma,
};
use proteus::{Retina, RetinaFormat};
use sha2::{Digest, Sha256};
use std::io::Cursor;
use std::process::Command;

fn sense(
    id: &str,
    pos: &str,
    words: &[(&str, usize)],
    parents: &[&str],
    gloss: &str,
) -> LexicalSense {
    LexicalSense {
        id: id.into(),
        pos: pos.into(),
        definition: gloss.into(),
        lemmas: words
            .iter()
            .map(|(word, count)| SenseLemma {
                text: (*word).into(),
                count: *count,
            })
            .collect(),
        parents: parents.iter().map(|parent| (*parent).into()).collect(),
        similar: Vec::new(),
        antonyms: Vec::new(),
    }
}

fn retina() -> Retina {
    let lexicon = Lexicon {
        schema: 1,
        source: LexiconSource {
            name: "Independent contextual fixture".into(),
            version: "1".into(),
            url: "urn:proteus:contextual-contract".into(),
            sha256: "0".repeat(64),
            license: "Synthetic fixture".into(),
        },
        senses: vec![
            sense("entity.n.01", "n", &[("entity", 1)], &[], "a thing"),
            sense(
                "finance.n.01",
                "n",
                &[("finance", 1)],
                &["entity.n.01"],
                "financial activity",
            ),
            sense(
                "lender.n.01",
                "n",
                &[("bank", 10), ("lender", 10)],
                &["finance.n.01"],
                "money finance loan",
            ),
            // Redundant ancestry path tests unique ancestor mass in a DAG.
            sense(
                "creditor.n.01",
                "n",
                &[("creditor", 1)],
                &["finance.n.01", "entity.n.01"],
                "money creditor",
            ),
            sense(
                "shore.n.01",
                "n",
                &[("bank", 10), ("shore", 10)],
                &["entity.n.01"],
                "bank river water shore",
            ),
            sense(
                "tilt.v.01",
                "v",
                &[("bank", 1000), ("tilt", 1)],
                &[],
                "slope tilt",
            ),
        ],
    };
    let text = "bank lender creditor shore finance entity tilt";
    let mut offset = 0;
    let tokens = text
        .split_whitespace()
        .enumerate()
        .map(|(position, word)| {
            let start = offset;
            offset += word.len() + 1;
            AnnotatedToken {
                start,
                end: start + word.len(),
                lemma: word.into(),
                pos: "NOUN".into(),
                dependency: "ROOT".into(),
                head: position,
                sentence: 0,
                morphology: Vec::new(),
            }
        })
        .collect();
    let document = AnnotatedDocument {
        id: "source".into(),
        source_start: 0,
        text: text.into(),
        tokens,
        parser: ParserProvenance {
            name: "Explicit annotations".into(),
            version: "1".into(),
            model: "fixture".into(),
            model_sha256: "1".repeat(64),
            source_sha256: "2".repeat(64),
            backend: "scalar-oracle".into(),
        },
    };
    let mut bytes = serde_json::to_vec(&document).unwrap();
    bytes.push(b'\n');
    let completion = serde_json::json!({"schema": 1, "complete": 1, "documents_sha256": format!("{:x}", Sha256::digest(&bytes))});
    bytes.extend(serde_json::to_vec(&completion).unwrap());
    bytes.push(b'\n');
    train_semantic(
        Cursor::new(bytes),
        lexicon,
        SemanticConfig {
            dimension: 16,
            min_count: 1,
            ..SemanticConfig::default()
        },
    )
    .unwrap()
    .0
}

fn usage(context: &str) -> ContextualWord {
    ContextualWord {
        word: "bank".into(),
        context: context.into(),
        part_of_speech: Some(PartOfSpeech::Noun),
        target: Some(ByteSpan { start: 0, end: 4 }),
        target_lemma: None,
    }
}

#[test]
fn explicit_pos_and_target_exclusion_recover_contextual_meaning() {
    let retina = retina();
    let model = retina.semantic_model().unwrap();
    let finance = model
        .disambiguate(&usage("bank money finance loan"))
        .unwrap();
    let river = model
        .disambiguate(&usage("bank river water shore"))
        .unwrap();
    assert_eq!(finance.selected_sense.as_deref(), Some("lender.n.01"));
    assert_eq!(river.selected_sense.as_deref(), Some("shore.n.01"));
    assert!(finance.senses.iter().all(|sense| sense.id != "tilt.v.01"));
    assert!(finance
        .senses
        .iter()
        .all(|sense| !sense.matching_terms.iter().any(|term| term == "bank")));
    let no_context = model.disambiguate(&usage("bank")).unwrap();
    assert!(!no_context.has_gloss_overlap);
    for sense in no_context.senses {
        assert!((sense.posterior - sense.prior).abs() < 1e-14);
    }
}

#[test]
fn bilateral_comparison_is_symmetric_and_exposes_distribution_ambiguity() {
    let retina = retina();
    let model = retina.semantic_model().unwrap();
    let first = usage("bank money finance loan");
    let second = usage("bank river water shore");
    let forward = model.compare_contexts(&first, &second).unwrap();
    let reverse = model.compare_contexts(&second, &first).unwrap();
    assert_eq!(forward.same_sense, Some(false));
    assert_eq!(forward.same_sense, reverse.same_sense);
    assert!(
        (forward.expected_similarity.unwrap() - reverse.expected_similarity.unwrap()).abs() < 1e-14
    );
    assert!(
        (forward.posterior_agreement.unwrap() - reverse.posterior_agreement.unwrap()).abs() < 1e-14
    );
    let same = model.compare_contexts(&first, &first).unwrap();
    assert_eq!(same.same_sense, Some(true));
    assert!(same.posterior_agreement.unwrap() < 1.0);
    let unknown = ContextualWord::new("unavailable", "money");
    let missing = model.compare_contexts(&first, &unknown).unwrap();
    assert_eq!(missing.same_sense, None);
    assert_eq!(missing.posterior_agreement, None);
    assert_eq!(
        missing.right.unavailable.as_deref(),
        Some("word_not_in_corpus")
    );
}

#[test]
fn information_content_matches_an_independent_subtree_mass_calculation() {
    let retina = retina();
    let model = retina.semantic_model().unwrap();
    // Five noun pseudo-occurrences; finance contains itself, lender and creditor.
    // Membership is specified explicitly, without production ancestor traversal.
    let mut total = 5.0;
    let mut finance = 3.0;
    let mut lender = 1.0;
    let mut creditor = 1.0;
    for entry in model.entries().values() {
        for evidence in &entry.senses {
            let weight = entry.count as f64 * evidence.prior;
            match evidence.sense.as_str() {
                "lender.n.01" => {
                    total += weight;
                    finance += weight;
                    lender += weight;
                }
                "creditor.n.01" => {
                    total += weight;
                    finance += weight;
                    creditor += weight;
                }
                "finance.n.01" => {
                    total += weight;
                    finance += weight;
                }
                "entity.n.01" | "shore.n.01" => total += weight,
                "tilt.v.01" => {}
                other => panic!("Unexpected independent fixture sense: {other}"),
            }
        }
    }
    let expected = 2.0 * (total / finance).ln() / ((total / lender).ln() + (total / creditor).ln());
    let actual = model
        .similarity("lender", "creditor", Relation::InformationSimilarity)
        .unwrap()
        .unwrap();
    assert!((expected - actual).abs() < 1e-13);
    assert_eq!(
        model
            .similarity("lender", "entity", Relation::InformationSimilarity)
            .unwrap(),
        Some(0.0)
    );
    assert_eq!(
        model
            .similarity("lender", "lender", Relation::InformationSimilarity)
            .unwrap(),
        Some(1.0)
    );
    assert_eq!(
        model
            .similarity("lender", "tilt", Relation::InformationSimilarity)
            .unwrap(),
        Some(0.0)
    );
}

#[test]
fn target_ranges_are_utf8_aligned_whole_words_and_observed_lemma_forms() {
    let retina = retina();
    let model = retina.semantic_model().unwrap();
    for span in [
        ByteSpan { start: 4, end: 5 },
        ByteSpan { start: 3, end: 10 },
        ByteSpan {
            start: 99,
            end: 100,
        },
    ] {
        let mut query = usage("café bank river");
        query.target = Some(span);
        assert!(model.disambiguate(&query).is_err());
    }
    let mut query = usage("café bank river");
    query.target = Some(ByteSpan { start: 6, end: 10 });
    assert_eq!(
        model
            .disambiguate(&query)
            .unwrap()
            .selected_sense
            .as_deref(),
        Some("shore.n.01")
    );
    query.target = Some(ByteSpan { start: 11, end: 16 });
    assert!(model.disambiguate(&query).is_err());
    let mut inflected = usage("banks money loan");
    inflected.target = Some(ByteSpan { start: 0, end: 5 });
    assert!(model.disambiguate(&inflected).is_err());
    inflected.target_lemma = Some("bank".into());
    assert_eq!(
        model
            .disambiguate(&inflected)
            .unwrap()
            .selected_sense
            .as_deref(),
        Some("lender.n.01")
    );
    inflected.target_lemma = Some("shore".into());
    assert!(model.disambiguate(&inflected).is_err());
    inflected.target = None;
    assert!(model.disambiguate(&inflected).is_err());
}

#[test]
fn evaluation_retains_unknown_and_invalid_observations_in_denominators() {
    let retina = retina();
    let model = retina.semantic_model().unwrap();
    let valid = ContextPair {
        id: "valid".into(),
        left: usage("bank money finance loan"),
        right: usage("bank river water shore"),
        same_sense: Some(false),
    };
    let unknown = ContextPair {
        id: "unknown".into(),
        left: usage("bank"),
        right: ContextualWord::new("unavailable", "river"),
        same_sense: Some(true),
    };
    let mut invalid = valid.clone();
    invalid.id = "invalid".into();
    invalid.left.target = Some(ByteSpan { start: 1, end: 3 });
    let report = evaluate_context_pairs(model, &[valid.clone(), unknown, invalid]).unwrap();
    assert_eq!(report.total_pairs, 3);
    assert_eq!(report.predictions.len(), 3);
    assert_eq!(report.candidate.available_pairs, 1);
    assert_eq!(report.candidate.accuracy, Some(1.0));
    assert_eq!(
        report.candidate.accuracy_with_unavailable_as_errors,
        Some(1.0 / 3.0)
    );
    assert!(!report.predictions[2].errors.is_empty());
    let line = serde_json::to_string(&valid).unwrap();
    assert!(read_context_pairs(Cursor::new(format!("{line}\n{line}\n"))).is_err());
    assert!(read_context_pairs(Cursor::new("{broken}\n")).is_err());
    assert_eq!(
        evaluate_context_pairs(model, &[])
            .unwrap()
            .candidate
            .accuracy,
        None
    );
}

#[test]
fn contextual_ranking_matches_independent_exhaustive_mixture_and_mapped_reload() {
    let retina = retina();
    let model = retina.semantic_model().unwrap();
    let query = usage("bank money finance loan");
    let prediction = model.disambiguate(&query).unwrap();
    let actual = model
        .find_similar_contextual(&query, Relation::InformationSimilarity, 100)
        .unwrap();
    let mut expected = Vec::new();
    for word in model
        .entries()
        .keys()
        .filter(|word| word.as_str() != "bank")
    {
        let mut score = 0.0;
        let mut available = false;
        for sense in &prediction.senses {
            if let Some(value) = model
                .score(
                    proteus::semantic::WordQuery {
                        word: "bank",
                        context: None,
                        sense: Some(&sense.id),
                    },
                    word,
                    Relation::InformationSimilarity,
                )
                .unwrap()
            {
                score += sense.posterior * value.score;
                available = true;
            }
        }
        if available {
            expected.push((word.clone(), score));
        }
    }
    expected.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
    assert_eq!(
        actual
            .iter()
            .map(|row| (row.word.clone(), row.score))
            .collect::<Vec<_>>(),
        expected
    );
    assert!(model
        .find_similar_contextual(&query, Relation::Association, 10)
        .is_err());
    assert!(model
        .find_similar_contextual(&query, Relation::Similarity, 0)
        .unwrap()
        .is_empty());
    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("first.retina");
    let second = directory.path().join("second.retina");
    retina.save(&first).unwrap();
    let loaded = Retina::load(&first).unwrap();
    let mapped = RetinaFormat::mmap(&first).unwrap();
    for other in [
        loaded.semantic_model().unwrap(),
        mapped.semantic_model().unwrap(),
    ] {
        assert_eq!(
            serde_json::to_value(other.disambiguate(&query).unwrap()).unwrap(),
            serde_json::to_value(&prediction).unwrap()
        );
        assert_eq!(
            other
                .similarity("lender", "creditor", Relation::InformationSimilarity)
                .unwrap(),
            model
                .similarity("lender", "creditor", Relation::InformationSimilarity)
                .unwrap()
        );
    }
    loaded.save(&second).unwrap();
    assert_eq!(
        std::fs::read(first).unwrap(),
        std::fs::read(second).unwrap()
    );
}

#[test]
fn native_cli_exposes_target_aware_queries_and_complete_context_evaluation() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.retina");
    retina().save(&path).unwrap();
    let binary = env!("CARGO_BIN_EXE_proteus");
    let result = Command::new(binary)
        .args(["semantic", "disambiguate", "-r"])
        .arg(&path)
        .args([
            "bank",
            "--context",
            "bank river water shore",
            "--pos",
            "n",
            "--target-start",
            "0",
            "--target-end",
            "4",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(value["selected_sense"], "shore.n.01");
    let failed = Command::new(binary)
        .args(["semantic", "disambiguate", "-r"])
        .arg(&path)
        .args(["bank", "--context", "bank river", "--target-start", "0"])
        .output()
        .unwrap();
    assert!(!failed.status.success());
    let pair = ContextPair {
        id: "probe".into(),
        left: usage("bank money finance loan"),
        right: usage("bank river water shore"),
        same_sense: Some(false),
    };
    let pairs = directory.path().join("pairs.jsonl");
    std::fs::write(
        &pairs,
        format!("{}\n", serde_json::to_string(&pair).unwrap()),
    )
    .unwrap();
    let output = Command::new(binary)
        .args(["semantic", "compare-contexts", "-r"])
        .arg(&path)
        .arg("--pairs")
        .arg(pairs)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["evaluation"]["candidate"]["correct"], 1);
}
