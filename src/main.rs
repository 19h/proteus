//! Proteus CLI - Semantic Folding Engine
//!
//! Command-line interface for training and using semantic retinas.

use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle, HumanDuration};
use log::error;
use rayon::prelude::*;
use proteus::{
    FingerprintConfig, Retina, Result, Som, SomConfig, SomTrainer, Tokenizer,
    WordFingerprinter, SegmentationConfig, SemanticSegmenter,
};
use proteus::som::training::{WordEmbeddings, DEFAULT_EMBEDDING_DIM};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "proteus")]
#[command(author = "Proteus Contributors")]
#[command(version)]
#[command(about = "Semantic Folding Engine", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a new retina from a corpus
    Train {
        /// Input corpus file (one document per line)
        #[arg(short, long)]
        input: PathBuf,

        /// Output retina file
        #[arg(short, long)]
        output: PathBuf,

        /// Grid dimension (default: 128)
        #[arg(short, long, default_value = "128")]
        dimension: usize,

        /// Number of training iterations
        #[arg(short = 'n', long, default_value = "100000")]
        iterations: usize,

        /// Random seed for reproducibility
        #[arg(short, long)]
        seed: Option<u64>,
    },

    /// Compute fingerprint for text
    Fingerprint {
        /// Retina file to use
        #[arg(short, long)]
        retina: PathBuf,

        /// Text to fingerprint
        text: String,

        /// Output format (positions, binary, hex)
        #[arg(short, long, default_value = "positions")]
        format: String,
    },

    /// Find similar words
    Similar {
        /// Retina file to use
        #[arg(short, long)]
        retina: PathBuf,

        /// Word to find similar words for
        word: String,

        /// Number of results
        #[arg(short = 'k', long, default_value = "10")]
        count: usize,
    },

    /// Compute similarity between texts
    Similarity {
        /// Retina file to use
        #[arg(short, long)]
        retina: PathBuf,

        /// First text
        text1: String,

        /// Second text
        text2: String,
    },

    /// Show retina statistics
    Info {
        /// Retina file to inspect
        retina: PathBuf,
    },

    /// Segment text into semantic sections
    Segment {
        /// Retina file to use
        #[arg(short, long)]
        retina: PathBuf,

        /// Input text file
        #[arg(short, long)]
        input: PathBuf,

        /// Sentences per block for comparison (default: 3)
        #[arg(short, long, default_value = "3")]
        block_size: usize,

        /// Depth threshold in std deviations (default: 0.5)
        #[arg(short = 't', long, default_value = "0.5")]
        threshold: f32,

        /// Minimum sentences per segment (default: 2)
        #[arg(short, long, default_value = "2")]
        min_segment: usize,

        /// Show similarity/depth scores for debugging
        #[arg(long)]
        debug: bool,
    },
}

fn main() {
    let cli = Cli::parse();

    // Initialize logging
    if cli.verbose {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    }

    let result = match cli.command {
        Commands::Train {
            input,
            output,
            dimension,
            iterations,
            seed,
        } => train_retina(input, output, dimension, iterations, seed),

        Commands::Fingerprint {
            retina,
            text,
            format,
        } => fingerprint_text(retina, text, format),

        Commands::Similar {
            retina,
            word,
            count,
        } => find_similar(retina, word, count),

        Commands::Similarity {
            retina,
            text1,
            text2,
        } => compute_similarity(retina, text1, text2),

        Commands::Info { retina } => show_info(retina),

        Commands::Segment {
            retina,
            input,
            block_size,
            threshold,
            min_segment,
            debug,
        } => segment_text(retina, input, block_size, threshold, min_segment, debug),
    };

    if let Err(e) = result {
        error!("Error: {}", e);
        std::process::exit(1);
    }
}

fn train_retina(
    input: PathBuf,
    output: PathBuf,
    dimension: usize,
    iterations: usize,
    seed: Option<u64>,
) -> Result<()> {
    let start_time = Instant::now();

    println!("Proteus Semantic Folding Engine");
    println!("   Training retina from: {}", input.display());
    println!();

    // Create progress bar style
    let spinner_style = ProgressStyle::default_spinner()
        .template("{spinner:.cyan} {msg}")
        .unwrap();

    let bar_style = ProgressStyle::default_bar()
        .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) ETA: {eta}")
        .unwrap()
        .progress_chars("█▓▒░  ");

    // Step 1: Load corpus
    let pb = ProgressBar::new_spinner();
    pb.set_style(spinner_style.clone());
    pb.set_message("Loading corpus...");
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let file = File::open(&input)?;
    let reader = BufReader::new(file);
    let documents: Vec<String> = reader.lines().map_while(|l| l.ok()).collect();

    pb.finish_and_clear();
    println!("✓ Loaded {} documents", format_number(documents.len()));

    // Step 2: Extract context windows (parallel)
    let pb = ProgressBar::new(documents.len() as u64);
    pb.set_style(bar_style.clone());
    pb.set_message("Extracting context windows (parallel)...");

    let tokenizer = Tokenizer::default_config();
    let contexts: Vec<(String, Vec<String>)> = documents
        .par_iter()
        .flat_map(|doc| {
            pb.inc(1);
            tokenizer.context_windows_as_strings(doc, 2)
        })
        .collect();

    pb.finish_and_clear();
    println!("✓ Extracted {} context windows", format_number(contexts.len()));

    // Step 3: Learn embeddings
    let pb = ProgressBar::new(0);
    pb.set_style(bar_style.clone());
    pb.set_message(format!("Learning word embeddings ({} dimensions)...", DEFAULT_EMBEDDING_DIM));

    let embeddings = WordEmbeddings::from_contexts_with_progress(&contexts, DEFAULT_EMBEDDING_DIM, seed, Some(&pb));

    pb.finish_and_clear();
    println!("✓ Learned embeddings for {} words", format_number(embeddings.len()));

    // Step 4: Initialize SOM
    let pb = ProgressBar::new_spinner();
    pb.set_style(spinner_style.clone());
    pb.set_message(format!("Initializing SOM ({}x{} = {} neurons)...", dimension, dimension, dimension * dimension));
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let som_config = SomConfig {
        dimension,
        weight_dimension: DEFAULT_EMBEDDING_DIM,
        iterations,
        seed,
        ..Default::default()
    };

    let mut som = Som::new(&som_config);
    let mut trainer = SomTrainer::new(som_config);

    pb.finish_and_clear();
    println!("✓ Initialized SOM ({}x{} = {} neurons, {}-dim weights)",
        dimension, dimension, dimension * dimension, DEFAULT_EMBEDDING_DIM);

    // Step 5: Train SOM with progress display
    println!();
    println!("Training SOM...");

    let word_to_bmus = trainer.train_fast_with_progress(&mut som, &embeddings, &contexts, |_, _, _, _| {})?;

    // Step 6: Generate fingerprints
    let pb = ProgressBar::new(word_to_bmus.len() as u64);
    pb.set_style(bar_style.clone());
    pb.set_message("Generating fingerprints...");

    let grid_size = (dimension * dimension) as u32;
    let fp_config = FingerprintConfig::default();
    let mut fingerprinter = WordFingerprinter::new(fp_config.clone(), grid_size);
    fingerprinter.create_fingerprints(&word_to_bmus, Some(&pb));

    pb.finish_and_clear();
    println!("✓ Created {} word fingerprints", format_number(fingerprinter.len()));

    // Step 7: Save retina
    let pb = ProgressBar::new_spinner();
    pb.set_style(spinner_style);
    pb.set_message("Saving retina...");
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let retina = Retina::with_index(
        fingerprinter.into_fingerprints(),
        dimension as u32,
        fp_config,
    );
    retina.save(&output)?;

    pb.finish_and_clear();
    println!("✓ Saved retina to {}", output.display());

    // Summary
    let elapsed = start_time.elapsed();
    println!();
    println!("Training complete in {}", HumanDuration(elapsed));
    println!("   Vocabulary: {} words", format_number(retina.vocabulary_size()));
    println!("   Grid: {}x{} ({} neurons)", dimension, dimension, dimension * dimension);
    println!("   Output: {}", output.display());

    Ok(())
}

/// Format large numbers with commas for readability
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn fingerprint_text(retina_path: PathBuf, text: String, format: String) -> Result<()> {
    let retina = Retina::load(&retina_path)?;
    let fp = retina.fingerprint_text(&text);

    match format.as_str() {
        "positions" => {
            let positions: Vec<u32> = fp.iter().collect();
            println!("{:?}", positions);
        }
        "binary" => {
            let dense = fp.to_dense();
            for (i, &bit) in dense.iter().enumerate() {
                if i > 0 && i % 128 == 0 {
                    println!();
                }
                print!("{}", bit);
            }
            println!();
        }
        "hex" => {
            let dense = fp.to_dense();
            for chunk in dense.chunks(8) {
                let byte: u8 = chunk
                    .iter()
                    .enumerate()
                    .fold(0u8, |acc, (i, &b)| acc | (b << i));
                print!("{:02x}", byte);
            }
            println!();
        }
        _ => {
            println!("Unknown format: {}", format);
        }
    }

    Ok(())
}

fn find_similar(retina_path: PathBuf, word: String, count: usize) -> Result<()> {
    let retina = Retina::load(&retina_path)?;

    if !retina.contains(&word) {
        println!("Word '{}' not found in vocabulary", word);
        return Ok(());
    }

    let similar = retina.find_similar_words(&word, count)?;

    println!("Words similar to '{}':", word);
    for (w, score) in similar {
        println!("  {:.4}  {}", score, w);
    }

    Ok(())
}

fn compute_similarity(retina_path: PathBuf, text1: String, text2: String) -> Result<()> {
    let retina = Retina::load(&retina_path)?;
    let similarity = retina.text_similarity(&text1, &text2);

    println!("Similarity: {:.4}", similarity);
    println!("Text 1: {}", text1);
    println!("Text 2: {}", text2);

    Ok(())
}

fn show_info(retina_path: PathBuf) -> Result<()> {
    let retina = Retina::load(&retina_path)?;

    println!("Retina: {:?}", retina_path);
    println!("  Grid dimension: {}", retina.dimension());
    println!("  Grid size: {}", retina.grid_size());
    println!("  Vocabulary size: {}", retina.vocabulary_size());

    Ok(())
}

fn segment_text(
    retina_path: PathBuf,
    input_path: PathBuf,
    block_size: usize,
    threshold: f32,
    min_segment: usize,
    debug: bool,
) -> Result<()> {
    // Load retina
    let retina = Retina::load(&retina_path)?;

    // Read input text
    let text = fs::read_to_string(&input_path)?;

    // Configure segmenter
    let config = SegmentationConfig {
        block_size,
        depth_threshold: threshold,
        min_segment_sentences: min_segment,
        smoothing: true,
        smoothing_sigma: 1.0,
    };

    // Create segmenter and segment text
    let mut segmenter = SemanticSegmenter::with_config(retina, config)?;
    let result = segmenter.segment(&text)?;

    // Show debug info if requested
    if debug {
        println!("=== Segmentation Analysis ===");
        println!("Sentences: {}", result.num_sentences);
        println!("Blocks: {}", result.num_blocks);
        println!(
            "Boundaries found: {} (at sentence indices: {:?})",
            result.boundary_indices.len(),
            result.boundary_indices
        );
        println!(
            "Mean depth: {:.4}, Std: {:.4}",
            result.depth_mean, result.depth_std
        );
        println!(
            "Threshold used: {:.2} std above mean = {:.4}",
            threshold,
            result.depth_mean + threshold * result.depth_std
        );

        if !result.similarity_scores.is_empty() {
            println!("\nSimilarity scores between blocks:");
            for (i, &sim) in result.similarity_scores.iter().enumerate() {
                println!("  Block {} <-> {}: {:.4}", i, i + 1, sim);
            }
        }

        if !result.depth_scores.is_empty() {
            println!("\nDepth scores:");
            for (i, &depth) in result.depth_scores.iter().enumerate() {
                let marker = if result.boundary_indices.contains(&((i + 1) * block_size)) {
                    " <-- BOUNDARY"
                } else {
                    ""
                };
                println!("  Gap {}: {:.4}{}", i, depth, marker);
            }
        }

        println!("\n=== Segments ===\n");
    }

    // Output segments
    for (i, segment) in result.segments.iter().enumerate() {
        if i > 0 {
            println!(); // Blank line between segments
        }
        println!("{}", segment.text);
    }

    Ok(())
}
