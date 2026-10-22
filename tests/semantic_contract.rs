//! Independent numerical and structural contracts for the grounded backend.
//! The lexical graph and dependency annotations below are explicit test oracles.

use proteus::evaluation::{evaluate_pairs, WordPair};
use proteus::semantic::{
    read_annotated_corpus, train_semantic, AnnotatedDocument, AnnotatedToken, LexicalSense,
    Lexicon, LexiconSource, ParserProvenance, Relation, SemanticConfig, SenseLemma, WordQuery,
};
use proteus::{Retina, RetinaFormat, RetinaHeader};
use sha2::{Digest, Sha256};
use std::io::Cursor;

fn lexical_sense(
    id: &str,
    pos: &str,
    lemmas: &[&str],
    parents: &[&str],
    definition: &str,
) -> LexicalSense {
    LexicalSense {
        id: id.into(),
        pos: pos.into(),
        definition: definition.into(),
        lemmas: lemmas
            .iter()
            .map(|word| SenseLemma {
                text: (*word).into(),
                count: 1,
            })
            .collect(),
        parents: parents.iter().map(|parent| (*parent).into()).collect(),
        similar: Vec::new(),
        antonyms: Vec::new(),
    }
}

fn lexicon() -> Lexicon {
    let mut senses = vec![
        lexical_sense("entity.n.01", "n", &["entity"], &[], "a thing"),
        lexical_sense(
            "animal.n.01",
            "n",
            &["animal"],
            &["entity.n.01"],
            "a living animal",
        ),
        lexical_sense(
            "dog.n.01",
            "n",
            &["dog", "hound"],
            &["animal.n.01"],
            "a domestic animal dog",
        ),
        lexical_sense(
            "cat.n.01",
            "n",
            &["cat"],
            &["animal.n.01"],
            "a domestic animal cat",
        ),
        lexical_sense(
            "ruler.n.01",
            "n",
            &["ruler"],
            &["entity.n.01"],
            "a person governing a nation",
        ),
        lexical_sense(
            "king.n.01",
            "n",
            &["king", "monarch"],
            &["ruler.n.01"],
            "a royal ruler",
        ),
        lexical_sense(
            "queen.n.01",
            "n",
            &["queen"],
            &["ruler.n.01"],
            "a royal ruler",
        ),
        lexical_sense(
            "bank.n.01",
            "n",
            &["bank", "lender"],
            &["entity.n.01"],
            "money finance loan business",
        ),
        lexical_sense(
            "bank.n.02",
            "n",
            &["bank", "shore"],
            &["entity.n.01"],
            "river water land shore",
        ),
        lexical_sense("city.n.01", "n", &["new york"], &["entity.n.01"], "a city"),
        lexical_sense(
            "cafe.n.01",
            "n",
            &["café"],
            &["entity.n.01"],
            "a coffee shop",
        ),
        lexical_sense(
            "chase.v.01",
            "v",
            &["chase", "pursue"],
            &[],
            "pursue an animal",
        ),
        lexical_sense("open.v.01", "v", &["open"], &[], "open something"),
        lexical_sense("close.v.01", "v", &["close"], &[], "close something"),
    ];
    senses[12].antonyms.push("close.v.01".into());
    Lexicon {
        schema: 1,
        source: LexiconSource {
            name: "Synthetic contract fixture".into(),
            version: "1".into(),
            url: "urn:proteus:semantic-contract".into(),
            sha256: "0".repeat(64),
            license: "Original synthetic test fixture".into(),
        },
        senses,
    }
}

struct TokenSpec<'a> {
    form: &'a str,
    lemma: &'a str,
    pos: &'a str,
    dependency: &'a str,
    head: usize,
}

fn token<'a>(
    form: &'a str,
    lemma: &'a str,
    pos: &'a str,
    dependency: &'a str,
    head: usize,
) -> TokenSpec<'a> {
    TokenSpec {
        form,
        lemma,
        pos,
        dependency,
        head,
    }
}

fn document(id: &str, specifications: &[TokenSpec<'_>]) -> AnnotatedDocument {
    let text = specifications
        .iter()
        .map(|token| token.form)
        .collect::<Vec<_>>()
        .join(" ");
    let mut offset = 0;
    let tokens = specifications
        .iter()
        .map(|specification| {
            let start = offset;
            let end = start + specification.form.len();
            offset = end + 1;
            AnnotatedToken {
                start,
                end,
                lemma: specification.lemma.into(),
                pos: specification.pos.into(),
                dependency: specification.dependency.into(),
                head: specification.head,
                sentence: 0,
                morphology: if matches!(specification.pos, "VERB" | "AUX") {
                    vec!["Tense=Pres".into(), "VerbForm=Fin".into()]
                } else {
                    Vec::new()
                },
            }
        })
        .collect();
    AnnotatedDocument {
        id: id.into(),
        source_start: 0,
        text,
        tokens,
        parser: ParserProvenance {
            name: "Explicit dependency oracle".into(),
            version: "1".into(),
            model: "synthetic".into(),
            model_sha256: "0".repeat(64),
            source_sha256: "1".repeat(64),
            backend: "synthetic-oracle".into(),
        },
    }
}

fn words(id: &str, text: &str) -> AnnotatedDocument {
    let specifications: Vec<_> = text
        .split_whitespace()
        .enumerate()
        .map(|(position, word)| token(word, word, "NOUN", "ROOT", position))
        .collect();
    document(id, &specifications)
}

fn event_documents() -> Vec<AnnotatedDocument> {
    vec![
        document("active", &[
            token("dogs", "dog", "NOUN", "nsubj", 1),
            token("chase", "chase", "VERB", "ROOT", 1),
            token("cats", "cat", "NOUN", "dobj", 1),
        ]),
        document("reverse", &[
            token("cats", "cat", "NOUN", "nsubj", 1),
            token("chase", "chase", "VERB", "ROOT", 1),
            token("dogs", "dog", "NOUN", "dobj", 1),
        ]),
        document("passive", &[
            token("cats", "cat", "NOUN", "nsubjpass", 2),
            token("are", "be", "AUX", "auxpass", 2),
            token("chased", "chase", "VERB", "ROOT", 2),
            token("by", "by", "ADP", "agent", 2),
            token("dogs", "dog", "NOUN", "pobj", 3),
        ]),
        document("negative", &[
            token("dogs", "dog", "NOUN", "nsubj", 3),
            token("do", "do", "AUX", "aux", 3),
            token("not", "not", "PART", "neg", 3),
            token("chase", "chase", "VERB", "ROOT", 3),
            token("cats", "cat", "NOUN", "dobj", 3),
        ]),
        words("lexical", "king monarch queen ruler animal dog hound cat bank lender shore open close mystery New York café"),
        words("finance", "bank money finance loan business"),
        words("river", "bank river water land shore"),
    ]
}

fn annotation_bytes(documents: &[AnnotatedDocument]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for document in documents {
        bytes.extend(serde_json::to_vec(document).unwrap());
        bytes.push(b'\n');
    }
    let footer = serde_json::json!({
        "schema": 1, "complete": documents.len(),
        "documents_sha256": format!("{:x}", Sha256::digest(&bytes)),
    });
    bytes.extend(serde_json::to_vec(&footer).unwrap());
    bytes.push(b'\n');
    bytes
}

fn train(documents: &[AnnotatedDocument]) -> Retina {
    train_semantic(
        Cursor::new(annotation_bytes(documents)),
        lexicon(),
        SemanticConfig {
            dimension: 16,
            min_count: 1,
            ..SemanticConfig::default()
        },
    )
    .unwrap()
    .0
}

#[test]
fn taxonomy_matches_independent_depth_calculation_and_relations_remain_distinct() {
    let retina = train(&event_documents());
    let model = retina.semantic_model().unwrap();
    // Root depth 1; ruler depth 2; king and queen depth 3.
    assert_eq!(
        model
            .similarity("king", "queen", Relation::Similarity)
            .unwrap(),
        Some(2.0 / 3.0)
    );
    assert_eq!(
        model
            .similarity("king", "monarch", Relation::Synonym)
            .unwrap(),
        Some(1.0)
    );
    assert_eq!(
        model
            .similarity("king", "ruler", Relation::Hypernym)
            .unwrap(),
        Some(0.5)
    );
    assert_eq!(
        model
            .similarity("ruler", "king", Relation::Hypernym)
            .unwrap(),
        Some(0.0)
    );
    assert_eq!(
        model
            .similarity("ruler", "king", Relation::Hyponym)
            .unwrap(),
        Some(0.5)
    );
    assert_eq!(
        model
            .similarity("open", "close", Relation::Antonym)
            .unwrap(),
        Some(1.0)
    );
    assert_eq!(
        model
            .similarity("open", "close", Relation::Similarity)
            .unwrap(),
        Some(0.0)
    );
    assert_eq!(
        model
            .similarity("king", "mystery", Relation::Similarity)
            .unwrap(),
        None
    );
}

#[test]
fn context_changes_the_sense_posterior_and_explicit_sense_selection_is_checked() {
    let retina = train(&event_documents());
    let model = retina.semantic_model().unwrap();
    let finance = model
        .explain_senses(WordQuery {
            word: "bank",
            context: Some("money finance loan business"),
            sense: None,
        })
        .unwrap();
    let river = model
        .explain_senses(WordQuery {
            word: "bank",
            context: Some("river water land shore"),
            sense: None,
        })
        .unwrap();
    assert_eq!(finance[0].id, "bank.n.01");
    assert_eq!(river[0].id, "bank.n.02");
    assert!(finance[0].posterior > finance[0].prior);
    assert!((finance.iter().map(|sense| sense.posterior).sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(model
        .explain_senses(WordQuery {
            word: "bank",
            context: None,
            sense: Some("king.n.01")
        })
        .is_err());
    let best = model
        .similarity("bank", "lender", Relation::Similarity)
        .unwrap()
        .unwrap();
    let expected = model
        .similarity("bank", "lender", Relation::ExpectedSimilarity)
        .unwrap()
        .unwrap();
    assert_eq!(best, 1.0);
    assert!(expected < best);
}

#[test]
fn bound_roles_passive_voice_and_explicit_negation_have_independent_outcomes() {
    let documents = event_documents();
    let retina = train(&documents);
    let model = retina.semantic_model().unwrap();
    let results = model
        .search_evidence(&documents[0], usize::MAX, 1.0)
        .unwrap();
    let score = |id: &str| results.iter().find(|result| result.document == id).unwrap();
    assert_eq!(score("active").score, 1.0);
    assert_eq!(score("passive").score, 1.0);
    assert_eq!(score("reverse").score, 0.0);
    assert_eq!(score("reverse").unknown, 1);
    assert_eq!(score("negative").score, -1.0);
    assert_eq!(score("negative").conflicted, 1);
    assert_eq!(score("finance").unknown, 1);
    assert_eq!(results.len(), documents.len());
    let negative_query = model
        .search_evidence(&documents[3], usize::MAX, 1.0)
        .unwrap();
    assert_eq!(
        negative_query
            .iter()
            .find(|result| result.document == "negative")
            .unwrap()
            .score,
        1.0
    );
    assert!(model.search_evidence(&documents[0], 3, f64::NAN).is_err());
}

#[test]
fn phrases_unicode_ranges_and_independent_document_support_survive_training() {
    let mut documents = event_documents();
    let mut duplicate = documents[1].clone();
    duplicate.id = "duplicate".into();
    documents.push(duplicate);
    let retina = train(&documents);
    assert!(retina.contains("NEW YORK"));
    assert!(retina.contains("CAFÉ"));
    assert_eq!(retina.word_similarity("New York", "new york").unwrap(), 1.0);
    let model = retina.semantic_model().unwrap();
    for entry in model.entries().values() {
        for sense in &entry.senses {
            for reference in &sense.examples {
                let source = documents
                    .iter()
                    .find(|document| document.id == reference.document)
                    .unwrap();
                assert!(source.text.get(reference.start..reference.end).is_some());
            }
        }
    }
    let unique = train(&event_documents());
    let duplicated_support = model.entries()["chase"].senses[0].supporting_documents;
    let original_support =
        unique.semantic_model().unwrap().entries()["chase"].senses[0].supporting_documents;
    assert_eq!(duplicated_support, original_support);
}

#[test]
fn v5_roundtrip_migration_mapped_ranking_and_exact_retraining_agree() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("semantic.retina");
    let second = directory.path().join("second.retina");
    let migrated = directory.path().join("migrated.retina");
    let retina = train(&event_documents());
    retina.save(&path).unwrap();
    train(&event_documents()).save(&second).unwrap();
    assert_eq!(
        std::fs::read(&path).unwrap(),
        std::fs::read(&second).unwrap()
    );
    let header = RetinaHeader::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(header.version, 5);
    let loaded = Retina::load(&path).unwrap();
    let mapped = RetinaFormat::mmap(&path).unwrap();
    assert!(!loaded.has_spatial_topology());
    for word in ["king", "bank", "New York"] {
        assert_eq!(
            retina.find_similar_words(word, 30).unwrap(),
            loaded.find_similar_words(word, 30).unwrap()
        );
        assert_eq!(
            loaded.find_similar_words(word, 30).unwrap(),
            mapped.find_similar(word, 30).unwrap()
        );
    }
    loaded.save(&migrated).unwrap();
    assert_eq!(
        std::fs::read(&path).unwrap(),
        std::fs::read(&migrated).unwrap()
    );
    let mut downgraded = std::fs::read(&path).unwrap();
    downgraded[4..6].copy_from_slice(&4_u16.to_le_bytes());
    std::fs::write(&second, downgraded).unwrap();
    assert!(Retina::load(&second).is_err());
    assert!(RetinaFormat::mmap(&second).is_err());
}

#[test]
fn unsupported_lexical_evidence_remains_missing_in_evaluation() {
    let retina = train(&event_documents());
    assert!(retina.contains("mystery"));
    let report = evaluate_pairs(
        &retina,
        &[
            WordPair {
                word1: "king".into(),
                word2: "monarch".into(),
                score: 9.0,
            },
            WordPair {
                word1: "mystery".into(),
                word2: "monarch".into(),
                score: 0.0,
            },
        ],
    )
    .unwrap();
    assert_eq!(report.evaluated_pairs, 1);
    assert_eq!(report.coverage, 0.5);
    assert_eq!(report.predictions[1].predicted, None);
    assert_eq!(report.spearman, None);
}

#[test]
fn malformed_graph_annotations_completion_and_metadata_are_rejected() {
    let documents = event_documents();
    let bytes = annotation_bytes(&documents);
    let footer_start = bytes[..bytes.len() - 1]
        .iter()
        .rposition(|&byte| byte == b'\n')
        .unwrap()
        + 1;
    assert!(read_annotated_corpus(Cursor::new(&bytes[..footer_start])).is_err());
    let mut corrupt = bytes.clone();
    corrupt[0] = b' ';
    assert!(read_annotated_corpus(Cursor::new(corrupt)).is_err());
    let mut cycle = documents[0].clone();
    cycle.tokens[0].head = 1;
    cycle.tokens[1].head = 0;
    assert!(cycle.validate().is_err());
    let mut invalid_range = words("unicode", "café");
    invalid_range.tokens[0].end -= 1;
    assert!(invalid_range.validate().is_err());
    let mut graph = lexicon();
    graph.senses[0].parents.push("king.n.01".into());
    assert!(Lexicon::read(Cursor::new(serde_json::to_vec(&graph).unwrap())).is_err());
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("existing.retina");
    let retina = train(&documents);
    retina.save(&path).unwrap();
    let original = std::fs::read(&path).unwrap();
    let mut bad_model = retina.semantic_model().unwrap().clone();
    let mut value = serde_json::to_value(&bad_model).unwrap();
    value["report"]["corpus_sha256"] = serde_json::Value::String("0".repeat(64));
    bad_model = serde_json::from_value(value).unwrap();
    assert!(bad_model.validate().is_err());
    let mut unknown_sense = serde_json::to_value(retina.semantic_model().unwrap()).unwrap();
    unknown_sense["entries"]["king"]["senses"][0]["sense"] =
        serde_json::Value::String("missing.n.01".into());
    let unknown_sense: proteus::semantic::SemanticModel =
        serde_json::from_value(unknown_sense).unwrap();
    assert!(unknown_sense
        .score(WordQuery::new("king"), "queen", Relation::Similarity)
        .is_err());
    let (header, fingerprints, mut metadata) = RetinaFormat::read_with_metadata(&path).unwrap();
    metadata.semantic_model = Some(bad_model);
    assert!(
        RetinaFormat::write_with_metadata(&path, &fingerprints, header.dimension, metadata)
            .is_err()
    );
    assert_eq!(std::fs::read(&path).unwrap(), original);
    let config = SemanticConfig {
        sense_temperature: f64::NAN,
        ..SemanticConfig::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn modal_tense_and_unsupported_negation_scope_do_not_become_positive_evidence() {
    let mut documents = event_documents();
    documents.push(document(
        "modal",
        &[
            token("dogs", "dog", "NOUN", "nsubj", 2),
            token("may", "may", "AUX", "aux", 2),
            token("chase", "chase", "VERB", "ROOT", 2),
            token("cats", "cat", "NOUN", "dobj", 2),
        ],
    ));
    let mut past = documents[0].clone();
    past.id = "past".into();
    past.tokens[1].morphology = vec!["Tense=Past".into(), "VerbForm=Fin".into()];
    documents.push(past);
    let scoped = document(
        "unsupported-scope",
        &[
            token("not", "not", "PART", "neg", 1),
            token("all", "all", "DET", "det", 2),
            token("dogs", "dog", "NOUN", "nsubj", 3),
            token("chase", "chase", "VERB", "ROOT", 3),
            token("cats", "cat", "NOUN", "dobj", 3),
        ],
    );
    documents.push(scoped.clone());
    let retina = train(&documents);
    let model = retina.semantic_model().unwrap();
    let results = model
        .search_evidence(&documents[0], usize::MAX, 1.0)
        .unwrap();
    for id in ["modal", "past", "unsupported-scope"] {
        let result = results.iter().find(|result| result.document == id).unwrap();
        assert_eq!(result.score, 0.0);
        assert_eq!(result.unknown, 1);
    }
    assert!(model.search_evidence(&scoped, 10, 1.0).is_err());
}

#[test]
fn one_letter_participants_survive_lexical_vocabulary_filtering() {
    let original = document(
        "xy",
        &[
            token("X", "X", "PROPN", "nsubj", 1),
            token("opens", "open", "VERB", "ROOT", 1),
            token("Y", "Y", "PROPN", "dobj", 1),
        ],
    );
    let reversed = document(
        "yx",
        &[
            token("Y", "Y", "PROPN", "nsubj", 1),
            token("opens", "open", "VERB", "ROOT", 1),
            token("X", "X", "PROPN", "dobj", 1),
        ],
    );
    let retina = train(&[original.clone(), reversed]);
    assert!(!retina.contains("X"));
    let results = retina
        .semantic_model()
        .unwrap()
        .search_evidence(&original, 2, 1.0)
        .unwrap();
    assert_eq!(results[0].document, "xy");
    assert_eq!(results[0].score, 1.0);
    assert_eq!(results[1].score, 0.0);
}

#[test]
fn compressed_metadata_rejects_excessive_lengths_and_corrupt_streams() {
    let directory = tempfile::tempdir().unwrap();
    let original = directory.path().join("original.retina");
    let corrupt = directory.path().join("corrupt.retina");
    train(&event_documents()).save(&original).unwrap();
    let original_bytes = std::fs::read(&original).unwrap();
    for claimed_length in [0_u64, u64::MAX] {
        let mut bytes = original_bytes.clone();
        bytes[64..72].copy_from_slice(&claimed_length.to_le_bytes());
        std::fs::write(&corrupt, bytes).unwrap();
        assert!(Retina::load(&corrupt).is_err());
        assert!(RetinaFormat::mmap(&corrupt).is_err());
    }
    let mut bytes = original_bytes;
    bytes[72] ^= 0xff;
    std::fs::write(&corrupt, bytes).unwrap();
    assert!(Retina::load(&corrupt).is_err());
    assert!(RetinaFormat::mmap(&corrupt).is_err());
}

#[test]
fn semantic_cli_wires_training_relation_search_and_invalid_input_atomically() {
    use std::process::Command;

    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("annotations.jsonl");
    let lexical = directory.path().join("lexicon.json");
    let output = directory.path().join("semantic.retina");
    let query = directory.path().join("query.jsonl");
    let documents = event_documents();
    std::fs::write(&input, annotation_bytes(&documents)).unwrap();
    std::fs::write(&lexical, serde_json::to_vec(&lexicon()).unwrap()).unwrap();
    std::fs::write(&query, annotation_bytes(&documents[..1])).unwrap();
    let trained = Command::new(env!("CARGO_BIN_EXE_proteus"))
        .args(["semantic", "train", "--min-count", "1", "-d", "16"])
        .arg("-i")
        .arg(&input)
        .arg("--lexicon")
        .arg(&lexical)
        .arg("-o")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        trained.status.success(),
        "{}",
        String::from_utf8_lossy(&trained.stderr)
    );
    let scored = Command::new(env!("CARGO_BIN_EXE_proteus"))
        .args(["semantic", "score", "-r"])
        .arg(&output)
        .args(["king", "monarch", "--relation", "synonym"])
        .output()
        .unwrap();
    assert!(scored.status.success());
    let value: serde_json::Value = serde_json::from_slice(&scored.stdout).unwrap();
    assert_eq!(value["score"], 1.0);
    let searched = Command::new(env!("CARGO_BIN_EXE_proteus"))
        .args(["semantic", "search", "-r"])
        .arg(&output)
        .arg("--query")
        .arg(&query)
        .args(["-k", "20"])
        .output()
        .unwrap();
    assert!(
        searched.status.success(),
        "{}",
        String::from_utf8_lossy(&searched.stderr)
    );
    let results: serde_json::Value = serde_json::from_slice(&searched.stdout).unwrap();
    assert_eq!(results[0]["document"], "active");
    assert_eq!(results[0]["score"], 1.0);
    let before = std::fs::read(&output).unwrap();
    std::fs::write(&input, b"{\"incomplete\":true}\n").unwrap();
    let failed = Command::new(env!("CARGO_BIN_EXE_proteus"))
        .args(["semantic", "train", "-i"])
        .arg(&input)
        .arg("--lexicon")
        .arg(&lexical)
        .arg("-o")
        .arg(&output)
        .output()
        .unwrap();
    assert!(!failed.status.success());
    assert_eq!(before, std::fs::read(&output).unwrap());
}
