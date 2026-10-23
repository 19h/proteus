//! Non-spatial distributional training and labelled quality evaluation.

use indicatif::{HumanDuration, ProgressBar, ProgressStyle};
use proteus::training::angular::{train_distributional, AngularConfig};
use proteus::training::DocumentMode;
use proteus::{FingerprintConfig, Result, Retina};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;
use std::time::Instant;

#[derive(serde::Serialize)]
struct EvaluationOutput<Prediction = proteus::evaluation::PairPrediction> {
    model_sha256: String,
    reference_sha256: String,
    score_kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pos_column: Option<usize>,
    #[serde(flatten)]
    evaluation: proteus::evaluation::EvaluationReport<Prediction>,
}

pub(crate) fn evaluate_retina(
    model_path: PathBuf,
    reference_path: PathBuf,
    score_column: usize,
    header: bool,
) -> Result<()> {
    evaluate_retina_relation(model_path, reference_path, score_column, header, None, None)
}

pub(crate) fn evaluate_retina_relation(
    model_path: PathBuf,
    reference_path: PathBuf,
    score_column: usize,
    header: bool,
    relation: Option<proteus::semantic::Relation>,
    pos_column: Option<usize>,
) -> Result<()> {
    if let Some(pos_column) = pos_column {
        return evaluate_retina_pos(
            model_path,
            reference_path,
            score_column,
            header,
            relation.ok_or_else(|| {
                proteus::ProteusError::Config(
                    "POS evaluation requires an explicit semantic relation".into(),
                )
            })?,
            pos_column,
        );
    }
    let model_sha256 = file_sha256(&model_path)?;
    let reference_sha256 = file_sha256(&reference_path)?;
    let retina = Retina::load(&model_path)?;
    let pairs = proteus::evaluation::read_word_pairs(
        BufReader::new(File::open(&reference_path)?),
        score_column,
        header,
    )?;
    let evaluation = if let Some(relation) = relation {
        let model = retina.semantic_model().ok_or_else(|| {
            proteus::ProteusError::Config(
                "Relation evaluation requires a knowledge-grounded retina".into(),
            )
        })?;
        proteus::evaluation::evaluate_with(&pairs, |left, right| {
            model.similarity(left, right, relation)
        })?
    } else {
        proteus::evaluation::evaluate_pairs(&retina, &pairs)?
    };
    let score_kind = if retina.semantic_model().is_some() {
        format!(
            "knowledge-grounded:{:?}",
            relation.unwrap_or(proteus::semantic::Relation::Similarity)
        )
    } else {
        "binary-cosine".into()
    };

    publish_evaluation(
        &model_path,
        &reference_path,
        EvaluationOutput {
            model_sha256,
            reference_sha256,
            score_kind,
            pos_column: None,
            evaluation,
        },
    )
}

fn evaluate_retina_pos(
    model_path: PathBuf,
    reference_path: PathBuf,
    score_column: usize,
    header: bool,
    relation: proteus::semantic::Relation,
    pos_column: usize,
) -> Result<()> {
    let model_sha256 = file_sha256(&model_path)?;
    let reference_sha256 = file_sha256(&reference_path)?;
    let observations = proteus::semantic::read_pos_word_pairs(
        BufReader::new(File::open(&reference_path)?),
        score_column,
        pos_column,
        header,
    )?;
    let retina = Retina::load(&model_path)?;
    let model = retina.semantic_model().ok_or_else(|| {
        proteus::ProteusError::Config("POS evaluation requires a knowledge-grounded retina".into())
    })?;
    let evaluation = proteus::semantic::evaluate_pos_pairs(model, &observations, relation)?;
    publish_evaluation(
        &model_path,
        &reference_path,
        EvaluationOutput {
            model_sha256,
            reference_sha256,
            score_kind: format!("knowledge-grounded:{relation:?}:pair-pos"),
            pos_column: Some(pos_column),
            evaluation,
        },
    )
}

fn publish_evaluation<Prediction: serde::Serialize>(
    model_path: &std::path::Path,
    reference_path: &std::path::Path,
    output: EvaluationOutput<Prediction>,
) -> Result<()> {
    if file_sha256(model_path)? != output.model_sha256
        || file_sha256(reference_path)? != output.reference_sha256
    {
        return Err(proteus::ProteusError::Config(
            "Evaluation inputs changed while being read".into(),
        ));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&output)
            .map_err(|error| proteus::ProteusError::Config(error.to_string()))?
    );
    Ok(())
}

pub(crate) fn file_sha256(path: &std::path::Path) -> Result<String> {
    use sha2::{Digest, Sha256};

    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        digest.update(&buffer[..bytes_read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// Corpus and angular encoding options; no SOM update parameters are accepted.
#[derive(clap::Args)]
pub(crate) struct TrainDistributionalArgs {
    /// UTF-8 training corpus
    #[arg(short, long)]
    input: PathBuf,
    /// Destination retina, written atomically
    #[arg(short, long)]
    output: PathBuf,
    /// Square display side; bit positions have no spatial adjacency
    #[arg(short, long, default_value_t = 128)]
    dimension: usize,
    /// Reproducible random seed
    #[arg(short, long, default_value_t = 42)]
    seed: u64,
    /// Document boundaries
    #[arg(long, value_parser = ["paragraphs", "lines"], default_value = "paragraphs")]
    documents: String,
    /// Maximum normalized tokens per context
    #[arg(long, default_value_t = 64)]
    snippet_tokens: usize,
    /// Minimum retained token frequency
    #[arg(long, default_value_t = 2)]
    min_count: usize,
    /// Maximum context-feature document frequency
    #[arg(long, default_value_t = 0.5)]
    max_document_frequency: f64,
    /// Dimension of the learned word-context subspace
    #[arg(long, default_value_t = 128)]
    embedding_dimension: usize,
    /// Orthogonal subspace iterations
    #[arg(long, default_value_t = 2)]
    power_iterations: usize,
    /// Preserve singular-direction scale instead of whitening spectral coordinates
    #[arg(long)]
    no_whitening: bool,
    /// Maximum token distance in local context counts
    #[arg(long, default_value_t = 5)]
    context_window: usize,
    /// Context marginal smoothing exponent
    #[arg(long, default_value_t = 0.75)]
    context_alpha: f64,
    /// Maximum active fraction of angular features
    #[arg(long, default_value_t = 0.02)]
    sparsity: f64,
    /// Optional additional cap on active bits
    #[arg(long)]
    max_active_bits: Option<usize>,
    /// Worker count; zero selects the Rayon default
    #[arg(long, default_value_t = 0)]
    threads: usize,
}

pub(crate) fn train_distributional_retina(arguments: TrainDistributionalArgs) -> Result<()> {
    let config = AngularConfig {
        dimension: arguments.dimension,
        document_mode: if arguments.documents == "lines" {
            DocumentMode::Lines
        } else {
            DocumentMode::Paragraphs
        },
        snippet_tokens: arguments.snippet_tokens,
        min_count: arguments.min_count,
        max_document_frequency: arguments.max_document_frequency,
        embedding_dimension: arguments.embedding_dimension,
        power_iterations: arguments.power_iterations,
        whiten: !arguments.no_whitening,
        context_window: arguments.context_window,
        context_alpha: arguments.context_alpha,
        seed: arguments.seed,
        threads: arguments.threads,
        fingerprint: FingerprintConfig {
            sparsity: arguments.sparsity,
            max_active_bits: arguments.max_active_bits.unwrap_or(usize::MAX),
            min_active_bits: 0,
            weighted_aggregation: true,
        },
        ..AngularConfig::default()
    };
    config.validate()?;
    let start = Instant::now();
    println!(
        "Training non-spatial distributional retina from {}",
        arguments.input.display()
    );
    let progress = ProgressBar::new(0);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise}] [{bar:40}] {pos}/{len}")
            .unwrap(),
    );
    progress.set_message("Encoding words");
    let (retina, report) = train_distributional(
        BufReader::new(File::open(&arguments.input)?),
        config,
        |completed, total| {
            progress.set_length(total as u64);
            progress.set_position(completed as u64);
        },
    )?;
    progress.finish_and_clear();
    retina.save(&arguments.output)?;
    println!(
        "Saved {} in {}",
        arguments.output.display(),
        HumanDuration(start.elapsed())
    );
    println!(
        "Words: {}; snippets: {}; features: {}; vector dimension: {}; bits per word: {}",
        report.vocabulary,
        report.snippets,
        report.features,
        report.vector_dimension,
        report.active_bits
    );
    println!(
        "Zero-context words omitted: {}; duplicate fingerprints: {}",
        report.zero_context_words, report.duplicate_fingerprints
    );
    Ok(())
}
