//! Native commands for knowledge-grounded training and relation-aware queries.

use proteus::semantic::{
    evaluate_context_pairs, read_annotated_corpus, read_context_pairs, train_semantic, ByteSpan,
    ContextualWord, Lexicon, PartOfSpeech, Relation, SemanticConfig, SemanticModel, WordQuery,
};
use proteus::{FingerprintConfig, ProteusError, Result, Retina};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

#[derive(clap::Subcommand)]
pub(crate) enum SemanticCommand {
    /// Train a v5 retina from complete parser annotations and an explicit lexicon
    Train(TrainSemanticArgs),
    /// Infer target-aware senses with optional byte-range exclusion and known POS
    Disambiguate {
        #[arg(short, long)]
        retina: PathBuf,
        word: String,
        #[command(flatten)]
        context: ContextArgs,
    },
    /// Rank retained words from one target-aware context posterior
    ContextualSimilar {
        #[arg(short, long)]
        retina: PathBuf,
        word: String,
        #[command(flatten)]
        context: ContextArgs,
        #[arg(long, default_value = "similarity")]
        relation: Relation,
        #[arg(short = 'k', long, default_value_t = 10)]
        count: usize,
    },
    /// Compare both usages and evaluate optional labels in complete JSONL pairs
    CompareContexts {
        #[arg(short, long)]
        retina: PathBuf,
        #[arg(long)]
        pairs: PathBuf,
    },
    /// Inspect sense priors, contextual posteriors and retained source references
    Senses {
        #[arg(short, long)]
        retina: PathBuf,
        word: String,
        #[arg(long)]
        context: Option<String>,
        #[arg(long)]
        sense: Option<String>,
    },
    /// Exhaustively rank corpus vocabulary under a declared semantic relation
    Similar {
        #[arg(short, long)]
        retina: PathBuf,
        word: String,
        #[arg(long, default_value = "similarity")]
        relation: Relation,
        #[arg(long)]
        context: Option<String>,
        #[arg(long)]
        sense: Option<String>,
        #[arg(short = 'k', long, default_value_t = 10)]
        count: usize,
    },
    /// Score a word pair, retaining unsupported lexical evidence as JSON null
    Score {
        #[arg(short, long)]
        retina: PathBuf,
        left: String,
        right: String,
        #[arg(long, default_value = "similarity")]
        relation: Relation,
        #[arg(long)]
        context: Option<String>,
        #[arg(long)]
        sense: Option<String>,
    },
    /// Search bound events with explicit support, contradiction and unknown
    Search {
        #[arg(short, long)]
        retina: PathBuf,
        /// Complete annotated query corpus containing exactly one document
        #[arg(short, long)]
        query: PathBuf,
        #[arg(short = 'k', long, default_value_t = 10)]
        count: usize,
        #[arg(long, default_value_t = 1.0)]
        conflict_penalty: f64,
    },
    /// Evaluate a specified relation against labelled pairs with explicit coverage
    Evaluate {
        #[arg(short, long)]
        retina: PathBuf,
        #[arg(long)]
        pairs: PathBuf,
        #[arg(long, default_value_t = 3)]
        score_column: usize,
        #[arg(long)]
        header: bool,
        #[arg(long, default_value = "similarity")]
        relation: Relation,
    },
    /// Inspect extracted event bindings and original text ranges
    Events {
        #[arg(short, long)]
        retina: PathBuf,
        #[arg(long)]
        document: Option<String>,
        #[arg(short = 'k', long, default_value_t = 20)]
        count: usize,
    },
}

#[derive(clap::Args)]
pub(crate) struct ContextArgs {
    #[arg(long)]
    context: String,
    #[arg(long)]
    pos: Option<PartOfSpeech>,
    #[arg(long, requires = "target_end")]
    target_start: Option<usize>,
    #[arg(long, requires = "target_start")]
    target_end: Option<usize>,
    #[arg(long, requires = "target_start")]
    target_lemma: Option<String>,
}

impl ContextArgs {
    fn query(self, word: String) -> ContextualWord {
        ContextualWord {
            word,
            context: self.context,
            part_of_speech: self.pos,
            target: self
                .target_start
                .zip(self.target_end)
                .map(|(start, end)| ByteSpan { start, end }),
            target_lemma: self.target_lemma,
        }
    }
}

#[derive(clap::Args)]
pub(crate) struct TrainSemanticArgs {
    /// Complete annotated JSONL corpus; '-' reads standard input
    #[arg(short, long)]
    input: PathBuf,
    /// Explicit lexical graph produced by prepare_semantic.py lexicon
    #[arg(long)]
    lexicon: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(short, long, default_value_t = 128)]
    dimension: usize,
    #[arg(long, default_value_t = 2)]
    min_count: usize,
    #[arg(long, default_value_t = 5)]
    context_window: usize,
    #[arg(long, default_value_t = 0.75)]
    context_alpha: f64,
    #[arg(long, default_value_t = 32)]
    max_associations: usize,
    #[arg(long, default_value_t = 3)]
    max_examples_per_sense: usize,
    #[arg(long, default_value_t = 5)]
    max_phrase_tokens: usize,
    #[arg(long, default_value_t = 5.0)]
    lexical_prior_strength: f64,
    #[arg(long, default_value_t = 0.5)]
    min_sense_support: f64,
    #[arg(long, default_value_t = 0.15)]
    sense_temperature: f64,
    #[arg(long, default_value_t = 0.02)]
    sparsity: f64,
    #[arg(long)]
    max_active_bits: Option<usize>,
}

pub(crate) fn run(command: SemanticCommand) -> Result<()> {
    match command {
        SemanticCommand::Train(arguments) => train(arguments),
        SemanticCommand::Disambiguate {
            retina,
            word,
            context,
        } => {
            let retina = Retina::load(retina)?;
            print_json(&model(&retina)?.disambiguate(&context.query(word))?)
        }
        SemanticCommand::ContextualSimilar {
            retina,
            word,
            context,
            relation,
            count,
        } => {
            let retina = Retina::load(retina)?;
            let query = context.query(word);
            print_json(&serde_json::json!({
                "query": model(&retina)?.disambiguate(&query)?,
                "relation": relation,
                "matches": model(&retina)?.find_similar_contextual(&query, relation, count)?,
            }))
        }
        SemanticCommand::CompareContexts { retina, pairs } => compare_contexts(retina, pairs),
        SemanticCommand::Senses {
            retina,
            word,
            context,
            sense,
        } => {
            let retina = Retina::load(retina)?;
            print_json(&model(&retina)?.explain_senses(WordQuery {
                word: &word,
                context: context.as_deref(),
                sense: sense.as_deref(),
            })?)
        }
        SemanticCommand::Similar {
            retina,
            word,
            relation,
            context,
            sense,
            count,
        } => {
            let retina = Retina::load(retina)?;
            print_json(&model(&retina)?.find_similar(
                WordQuery {
                    word: &word,
                    context: context.as_deref(),
                    sense: sense.as_deref(),
                },
                relation,
                count,
            )?)
        }
        SemanticCommand::Score {
            retina,
            left,
            right,
            relation,
            context,
            sense,
        } => {
            let retina = Retina::load(retina)?;
            print_json(&model(&retina)?.score(
                WordQuery {
                    word: &left,
                    context: context.as_deref(),
                    sense: sense.as_deref(),
                },
                &right,
                relation,
            )?)
        }
        SemanticCommand::Search {
            retina,
            query,
            count,
            conflict_penalty,
        } => {
            let retina = Retina::load(retina)?;
            let documents = read_annotated_corpus(input_reader(query)?)?;
            if documents.len() != 1 {
                return Err(ProteusError::Config(
                    "An evidence query requires exactly one annotated document".into(),
                ));
            }
            print_json(&model(&retina)?.search_evidence(&documents[0], count, conflict_penalty)?)
        }
        SemanticCommand::Evaluate {
            retina,
            pairs,
            score_column,
            header,
            relation,
        } => super::quality::evaluate_retina_relation(
            retina,
            pairs,
            score_column,
            header,
            Some(relation),
        ),
        SemanticCommand::Events {
            retina,
            document,
            count,
        } => {
            let retina = Retina::load(retina)?;
            let events: Vec<_> = model(&retina)?
                .events()
                .iter()
                .filter(|event| document.as_deref().is_none_or(|id| id == event.document))
                .take(count)
                .collect();
            print_json(&events)
        }
    }
}

fn compare_contexts(retina_path: PathBuf, pairs_path: PathBuf) -> Result<()> {
    let model_sha256 = super::quality::file_sha256(&retina_path)?;
    let pairs_sha256 = super::quality::file_sha256(&pairs_path)?;
    let retina = Retina::load(&retina_path)?;
    let pairs = read_context_pairs(BufReader::new(File::open(&pairs_path)?))?;
    let result = evaluate_context_pairs(model(&retina)?, &pairs)?;
    if model_sha256 != super::quality::file_sha256(&retina_path)?
        || pairs_sha256 != super::quality::file_sha256(&pairs_path)?
    {
        return Err(ProteusError::Config(
            "Context evaluation inputs changed while being read".into(),
        ));
    }
    print_json(
        &serde_json::json!({ "model_sha256": model_sha256, "pairs_sha256": pairs_sha256, "evaluation": result }),
    )
}

fn model(retina: &Retina) -> Result<&SemanticModel> {
    retina.semantic_model().ok_or_else(|| {
        ProteusError::Config("This command requires a v5 knowledge-grounded retina".into())
    })
}

fn input_reader(path: PathBuf) -> Result<Box<dyn BufRead>> {
    if path.as_os_str() == "-" {
        Ok(Box::new(BufReader::new(std::io::stdin())))
    } else {
        Ok(Box::new(BufReader::new(File::open(path)?)))
    }
}

fn print_json(value: &impl serde::Serialize) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(value)
            .map_err(|error| ProteusError::Serialization(error.to_string()))?
    );
    Ok(())
}

fn train(arguments: TrainSemanticArgs) -> Result<()> {
    let config = SemanticConfig {
        dimension: arguments.dimension,
        min_count: arguments.min_count,
        context_window: arguments.context_window,
        context_alpha: arguments.context_alpha,
        max_associations: arguments.max_associations,
        max_examples_per_sense: arguments.max_examples_per_sense,
        max_phrase_tokens: arguments.max_phrase_tokens,
        lexical_prior_strength: arguments.lexical_prior_strength,
        min_sense_support: arguments.min_sense_support,
        sense_temperature: arguments.sense_temperature,
        fingerprint: FingerprintConfig {
            sparsity: arguments.sparsity,
            max_active_bits: arguments.max_active_bits.unwrap_or(usize::MAX),
            min_active_bits: 0,
            weighted_aggregation: true,
        },
        ..SemanticConfig::default()
    };
    config.validate()?;
    let lexicon = Lexicon::read(BufReader::new(File::open(arguments.lexicon)?))?;
    let (retina, report) = train_semantic(input_reader(arguments.input)?, lexicon, config)?;
    retina.save(arguments.output)?;
    print_json(&report)
}
