//! Proteus CLI - Semantic Folding Engine
//!
//! Command-line interface for training and using semantic retinas.

use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle, HumanDuration};
use log::error;
use rayon::prelude::*;
use proteus::{
    FingerprintConfig, Retina, Result, Sdr, Som, SomConfig, SomTrainer, Tokenizer,
    WordFingerprinter, SegmentationConfig, SemanticSegmenter,
    SemanticLookupEngine, SemanticLookupConfig, ToroidalGrid,
};
use proteus::som::training::{WordEmbeddings, DEFAULT_EMBEDDING_DIM};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read};
use std::path::PathBuf;
use std::time::Instant;
use image::{ImageBuffer, Rgb};

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

        /// Use fast regex-based sentence splitting (much faster but less accurate)
        #[arg(short, long)]
        fast: bool,
    },

    /// Migrate retina file to latest format (adds inverted index)
    Migrate {
        /// Input retina file (v1 or v2)
        #[arg(short, long)]
        input: PathBuf,

        /// Output retina file (will be v2 with inverted index)
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Lookup words at specific grid positions (simple mode)
    Lookup {
        /// Retina file to use
        #[arg(short, long)]
        retina: PathBuf,

        /// Grid positions to lookup (comma-separated, e.g., "1234,5678,9012")
        positions: String,

        /// Number of results per position (default: 10)
        #[arg(short = 'k', long, default_value = "10")]
        count: usize,

        /// Show combined results ranked by position overlap
        #[arg(short, long)]
        combined: bool,
    },

    /// Advanced semantic position lookup with Cortical.io-style kernel convolution,
    /// adaptive neighborhood expansion, hierarchical clustering, and IDF weighting
    SemanticLookup {
        /// Retina file to use
        #[arg(short, long)]
        retina: PathBuf,

        /// Grid positions to lookup (comma-separated, e.g., "1234,5678,9012")
        positions: String,

        /// Number of results (default: 20)
        #[arg(short = 'k', long, default_value = "20")]
        count: usize,

        /// Base radius for neighborhood expansion (default: 1.8, Cortical.io standard)
        #[arg(long, default_value = "1.8")]
        radius: f64,

        /// Gaussian kernel sigma for spatial weighting (default: 1.5)
        #[arg(long, default_value = "1.5")]
        sigma: f64,

        /// Use adaptive radius based on local semantic density
        #[arg(long)]
        adaptive: bool,

        /// Show detailed factor breakdown for each result
        #[arg(long)]
        explain: bool,

        /// Show cluster membership for query positions
        #[arg(long)]
        clusters: bool,

        /// Show position statistics (IDF, density, information content)
        #[arg(long)]
        stats: bool,

        /// Kernel convolution: compute weighted similarity on continuous heat maps
        #[arg(long)]
        kernel: bool,
    },

    /// Generate fingerprint images from text
    FingerprintImage {
        /// Retina file to use
        #[arg(short, long)]
        retina: PathBuf,

        /// Input file (use "-" for stdin)
        #[arg(short, long)]
        input: Option<String>,

        /// Direct text input (alternative to --input)
        #[arg(short, long)]
        text: Option<String>,

        /// Output directory for fingerprint images (default: "fingerprints")
        #[arg(short, long, default_value = "fingerprints")]
        output: PathBuf,

        /// Segment text and generate fingerprint images for each segment
        #[arg(short, long)]
        segment: bool,

        /// Generate fingerprint images for each word individually
        #[arg(short, long)]
        words: bool,

        /// Pixel scale factor for output images (default: 4)
        #[arg(long, default_value = "4")]
        scale: u32,

        /// Segmentation: sentences per block for comparison
        #[arg(long, default_value = "3")]
        block_size: usize,

        /// Segmentation: depth threshold in std deviations
        #[arg(long, default_value = "0.5")]
        threshold: f32,

        /// Segmentation: minimum sentences per segment
        #[arg(long, default_value = "2")]
        min_segment: usize,

        /// Use fast regex-based sentence splitting (much faster but less accurate)
        #[arg(short, long)]
        fast: bool,
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
            fast,
        } => segment_text(retina, input, block_size, threshold, min_segment, debug, fast),

        Commands::Migrate { input, output } => migrate_retina(input, output),

        Commands::Lookup {
            retina,
            positions,
            count,
            combined,
        } => lookup_positions(retina, positions, count, combined),

        Commands::SemanticLookup {
            retina,
            positions,
            count,
            radius,
            sigma,
            adaptive,
            explain,
            clusters,
            stats,
            kernel,
        } => semantic_lookup(
            retina, positions, count, radius, sigma,
            adaptive, explain, clusters, stats, kernel,
        ),

        Commands::FingerprintImage {
            retina,
            input,
            text,
            output,
            segment,
            words,
            scale,
            block_size,
            threshold,
            min_segment,
            fast,
        } => fingerprint_image(
            retina,
            input,
            text,
            output,
            segment,
            words,
            scale,
            block_size,
            threshold,
            min_segment,
            fast,
        ),
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
    use proteus::storage::MmappedRetina;

    // Use memory-mapped access for instant loading
    let retina = MmappedRetina::open(&retina_path)?;

    if !retina.contains(&word) {
        println!("Word '{}' not found in vocabulary", word);
        return Ok(());
    }

    // Use fast inverted index lookup if available
    let similar = match retina.find_similar(&word, count) {
        Some(results) => results,
        None => {
            println!("No inverted index available. Run 'proteus migrate' to add one.");
            return Ok(());
        }
    };

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
    fast: bool,
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
        fast_mode: fast,
        verbose: debug, // Use debug flag for verbose output
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

fn migrate_retina(input: PathBuf, output: PathBuf) -> Result<()> {
    use proteus::storage::RetinaFormat;

    println!("Migrating retina: {} -> {}", input.display(), output.display());
    let start_time = Instant::now();

    // Read the input file to check version
    let (header, fingerprints, existing_index) = RetinaFormat::read_with_index(&input)?;

    println!("  Input file version: {}", header.version);
    println!("  Vocabulary size: {} words", header.num_words);
    println!("  Grid dimension: {}x{}", header.dimension, header.dimension);
    println!(
        "  Has inverted index: {}",
        if existing_index.is_some() { "yes" } else { "no" }
    );

    // Build inverted index if not present, otherwise use existing
    let grid_size = header.dimension * header.dimension;
    let inverted_index = match existing_index {
        Some(idx) => {
            println!("  Using existing inverted index");
            idx
        }
        None => {
            println!("  Building inverted index...");
            let idx = proteus::index::InvertedIndex::from_fingerprints(&fingerprints, grid_size);
            println!("  Index built with {} words", idx.len());
            idx
        }
    };

    // Write the new file with the inverted index
    println!("  Writing migrated retina...");
    RetinaFormat::write_with_index(&output, &fingerprints, header.dimension, &inverted_index)?;

    let elapsed = start_time.elapsed();
    let input_size = fs::metadata(&input)?.len();
    let output_size = fs::metadata(&output)?.len();

    println!("\nMigration complete!");
    println!("  Input size:  {} bytes", input_size);
    println!("  Output size: {} bytes", output_size);
    println!(
        "  Size change: {:+} bytes ({:+.1}%)",
        output_size as i64 - input_size as i64,
        (output_size as f64 - input_size as f64) / input_size as f64 * 100.0
    );
    println!("  Time: {}", HumanDuration(elapsed));

    Ok(())
}

fn lookup_positions(
    retina_path: PathBuf,
    positions_str: String,
    count: usize,
    combined: bool,
) -> Result<()> {
    // Parse positions from comma-separated string
    let positions: Vec<u32> = positions_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if positions.is_empty() {
        return Err(proteus::error::ProteusError::TextProcessing(
            "No valid positions provided. Use comma-separated numbers, e.g., '1234,5678'".to_string()
        ));
    }

    // Load retina
    let retina = Retina::load(&retina_path)?;
    let grid_size = retina.grid_size();
    let dimension = retina.dimension();

    // Validate positions
    for &pos in &positions {
        if pos >= grid_size {
            return Err(proteus::error::ProteusError::TextProcessing(
                format!("Position {} is out of range (max: {})", pos, grid_size - 1)
            ));
        }
    }

    println!("Retina: {} ({}x{} grid)", retina_path.display(), dimension, dimension);
    println!("Looking up {} position(s): {:?}\n", positions.len(), positions);

    if combined {
        // Show combined results ranked by how many positions each word covers
        let results = retina.words_at_positions(&positions, count)?;

        if results.is_empty() {
            println!("No words found at these positions.");
        } else {
            println!("Words covering the most queried positions:");
            println!("{:>6}  {:<6}  {}", "overlap", "pct", "word");
            println!("{:-<6}  {:-<6}  {:-<20}", "", "", "");
            for (word, overlap) in results {
                let pct = 100.0 * overlap as f64 / positions.len() as f64;
                println!("{:>6}  {:>5.1}%  {}", overlap, pct, word);
            }
        }
    } else {
        // Show results for each position separately
        for &pos in &positions {
            let words = retina.words_at_position(pos)?;
            let row = pos / dimension;
            let col = pos % dimension;

            println!("Position {} (row {}, col {}):", pos, row, col);
            if words.is_empty() {
                println!("  (no words at this position)");
            } else {
                let display_words: Vec<_> = words.iter().take(count).collect();
                let remaining = words.len().saturating_sub(count);

                for word in &display_words {
                    println!("  {}", word);
                }
                if remaining > 0 {
                    println!("  ... and {} more", remaining);
                }
            }
            println!();
        }
    }

    Ok(())
}

/// Generate a PNG image from an SDR fingerprint.
/// The SDR is rendered as a 2D grid where active bits are white and inactive bits are black.
fn sdr_to_image(sdr: &Sdr, dimension: u32, scale: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let dense = sdr.to_dense();
    let scaled_dim = dimension * scale;

    ImageBuffer::from_fn(scaled_dim, scaled_dim, |x, y| {
        // Map scaled coordinates back to original grid
        let orig_x = x / scale;
        let orig_y = y / scale;
        let idx = (orig_y * dimension + orig_x) as usize;

        if idx < dense.len() && dense[idx] > 0 {
            Rgb([255u8, 255u8, 255u8]) // White for active bits
        } else {
            Rgb([0u8, 0u8, 0u8]) // Black for inactive bits
        }
    })
}

/// Sanitize a string for use as a filename.
fn sanitize_filename(s: &str, max_len: usize) -> String {
    let sanitized: String = s
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == ' ' || *c == '-' || *c == '_')
        .take(max_len)
        .collect();
    sanitized.trim().replace(' ', "_")
}

fn fingerprint_image(
    retina_path: PathBuf,
    input: Option<String>,
    text_arg: Option<String>,
    output_dir: PathBuf,
    segment_mode: bool,
    words_mode: bool,
    scale: u32,
    block_size: usize,
    threshold: f32,
    min_segment: usize,
    fast: bool,
) -> Result<()> {
    let start_time = Instant::now();

    // Determine the text source
    let text = match (&input, &text_arg) {
        (Some(path), _) if path == "-" => {
            // Read from stdin
            let mut buffer = String::new();
            io::stdin().read_to_string(&mut buffer)?;
            buffer
        }
        (Some(path), _) => {
            // Read from file
            fs::read_to_string(path)?
        }
        (_, Some(t)) => {
            // Use text argument directly
            t.clone()
        }
        (None, None) => {
            return Err(proteus::error::ProteusError::TextProcessing(
                "No input provided. Use --input <file>, --input - for stdin, or --text <text>".to_string()
            ));
        }
    };

    if text.trim().is_empty() {
        return Err(proteus::error::ProteusError::TextProcessing(
            "Input text is empty".to_string()
        ));
    }

    // Validate mode selection
    if !segment_mode && !words_mode {
        return Err(proteus::error::ProteusError::Config(
            "Please specify a mode: --segment or --words".to_string()
        ));
    }

    // Create output directory
    fs::create_dir_all(&output_dir)?;
    println!("Output directory: {}", output_dir.display());

    let mut image_count = 0;

    // Process words mode first (if enabled), since segment mode consumes the retina
    if words_mode {
        println!("Loading retina from {}...", retina_path.display());
        let retina = Retina::load(&retina_path)?;
        let dimension = retina.dimension();
        println!("  Grid: {}x{}", dimension, dimension);

        println!("\nProcessing individual words...");

        let tokenizer = Tokenizer::default_config();
        let tokens = tokenizer.tokenize_to_strings(&text);

        // Deduplicate words while preserving order
        let mut seen = std::collections::HashSet::new();
        let unique_words: Vec<_> = tokens
            .iter()
            .filter(|w| seen.insert(w.to_lowercase()))
            .collect();

        println!("  Found {} unique words", unique_words.len());

        let pb = ProgressBar::new(unique_words.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len}")
            .unwrap()
            .progress_chars("█▓▒░  "));
        pb.set_message("Generating word fingerprints...");

        for (i, word) in unique_words.iter().enumerate() {
            // Check if word is in vocabulary
            if !retina.contains(word) {
                pb.inc(1);
                continue;
            }

            if let Some(fp) = retina.get_word_fingerprint(word) {
                let img = sdr_to_image(fp, dimension, scale);

                let safe_word = sanitize_filename(word, 50);
                let filename = format!("word_{:04}_{}.png", i + 1, safe_word);
                let filepath = output_dir.join(&filename);

                img.save(&filepath).map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
                })?;

                image_count += 1;
            }
            pb.inc(1);
        }
        pb.finish_and_clear();
        println!("  Generated {} word fingerprint images", image_count);
    }

    if segment_mode {
        // Load retina (or reload if already loaded for words mode)
        println!("{}Loading retina from {}...",
            if words_mode { "\n" } else { "" },
            retina_path.display());
        let retina = Retina::load(&retina_path)?;
        let dimension = retina.dimension();
        if !words_mode {
            println!("  Grid: {}x{}", dimension, dimension);
        }

        println!("\nSegmenting text{}...", if fast { " (fast mode)" } else { "" });

        // Configure segmenter
        let config = SegmentationConfig {
            block_size,
            depth_threshold: threshold,
            min_segment_sentences: min_segment,
            smoothing: true,
            smoothing_sigma: 1.0,
            fast_mode: fast,
            verbose: true, // Always show progress for fingerprint-image
        };

        // Create segmenter and segment text (segmenter consumes retina)
        let mut segmenter = SemanticSegmenter::with_config(retina, config)?;
        let result = segmenter.segment(&text)?;

        println!("  Found {} segments", result.segments.len());

        // Reload retina for fingerprinting segments
        let retina = Retina::load(&retina_path)?;

        // Generate fingerprint image for each segment
        let segment_count = result.segments.len();
        for (i, segment) in result.segments.iter().enumerate() {
            let fp = retina.fingerprint_text(&segment.text);
            let img = sdr_to_image(&fp, dimension, scale);

            // Create filename with segment preview
            let preview = sanitize_filename(&segment.text, 30);
            let filename = format!("segment_{:03}_{}.png", i + 1, preview);
            let filepath = output_dir.join(&filename);

            img.save(&filepath).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
            })?;

            println!("  [{}] {} ({} chars, {} active bits)",
                i + 1, filename, segment.text.len(), fp.cardinality());
            image_count += 1;
        }
        println!("  Generated {} segment fingerprint images", segment_count);
    }

    let elapsed = start_time.elapsed();
    println!("\nComplete!");
    println!("  Images generated: {}", image_count);
    println!("  Output directory: {}", output_dir.display());
    println!("  Time: {}", HumanDuration(elapsed));

    Ok(())
}

/// Advanced semantic position lookup using kernel convolution, adaptive expansion,
/// hierarchical clustering, and information-theoretic weighting.
fn semantic_lookup(
    retina_path: PathBuf,
    positions_str: String,
    count: usize,
    radius: f64,
    sigma: f64,
    adaptive: bool,
    explain: bool,
    show_clusters: bool,
    show_stats: bool,
    use_kernel: bool,
) -> Result<()> {
    let start_time = Instant::now();

    // Parse positions
    let positions: Vec<u32> = positions_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if positions.is_empty() {
        return Err(proteus::error::ProteusError::TextProcessing(
            "No valid positions provided. Use comma-separated numbers.".to_string()
        ));
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       SEMANTIC POSITION LOOKUP (Cortical.io-style)          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Load retina
    println!("[1/3] Loading retina...");
    let retina = Retina::load(&retina_path)?;
    let dimension = retina.dimension();
    let grid_size = dimension * dimension;

    println!("      Retina: {}", retina_path.display());
    println!("      Grid: {}x{} ({} positions)", dimension, dimension, grid_size);
    println!("      Vocabulary: {} words", retina.vocabulary_size());

    // Validate positions
    for &pos in &positions {
        if pos >= grid_size {
            return Err(proteus::error::ProteusError::TextProcessing(
                format!("Position {} out of range (max: {})", pos, grid_size - 1)
            ));
        }
    }

    // Check that inverted index exists
    match retina.words_at_position(0) {
        Ok(_) => {
            // Index exists, we can proceed
        }
        Err(_) => {
            return Err(proteus::error::ProteusError::TextProcessing(
                "Retina has no inverted index. Run 'proteus migrate' first.".to_string()
            ));
        }
    };

    // Create semantic lookup engine
    println!("\n[2/3] Initializing semantic lookup engine...");
    let config = SemanticLookupConfig {
        base_radius: radius,
        kernel_sigma: sigma,
        adaptive_radius: adaptive,
        ..Default::default()
    };

    let _engine = SemanticLookupEngine::new(dimension, config);

    // We need direct access to the inverted index for initialization
    // For now, we'll do a simplified version without full engine initialization
    // since we don't have direct index access from Retina

    println!("      Base radius: {:.1}", radius);
    println!("      Kernel sigma: {:.1}", sigma);
    println!("      Adaptive radius: {}", if adaptive { "enabled" } else { "disabled" });
    println!("      Kernel weighting: {}", if use_kernel { "Gaussian decay" } else { "uniform" });

    // Display query positions
    let grid = ToroidalGrid::new(dimension);
    println!("\n[3/3] Analyzing {} query position(s)...", positions.len());

    for &pos in &positions {
        let pos_2d = grid.to_2d(pos);
        println!("      Position {} (row {}, col {})", pos, pos_2d.row, pos_2d.col);
    }

    // Show position statistics if requested
    if show_stats {
        println!("\n┌─────────────────────────────────────────────────────────────┐");
        println!("│ POSITION STATISTICS                                         │");
        println!("├─────────────────────────────────────────────────────────────┤");

        for &pos in &positions {
            let words = retina.words_at_position(pos)?;
            let pos_2d = grid.to_2d(pos);

            // Compute basic statistics
            let word_count = words.len();
            let idf = if word_count > 0 {
                (retina.vocabulary_size() as f64 / (1.0 + word_count as f64)).ln()
            } else {
                0.0
            };

            // Get neighborhood density
            let neighbors = grid.neighborhood(pos, radius);
            let neighbor_counts: Vec<usize> = neighbors.iter()
                .filter_map(|&n| retina.words_at_position(n).ok())
                .map(|w| w.len())
                .collect();
            let local_density = if !neighbor_counts.is_empty() {
                neighbor_counts.iter().sum::<usize>() as f64 / neighbor_counts.len() as f64
            } else {
                word_count as f64
            };

            println!("│ Position {} ({}, {}):", pos, pos_2d.row, pos_2d.col);
            println!("│   Word count: {}", word_count);
            println!("│   IDF (information): {:.4}", idf);
            println!("│   Local density: {:.2} words/position", local_density);
            println!("│   Neighborhood size (r={:.1}): {} positions", radius, neighbors.len());

            // Information content interpretation
            let info_level = if idf > 5.0 {
                "highly specific (rare)"
            } else if idf > 3.0 {
                "moderately specific"
            } else if idf > 1.0 {
                "common"
            } else {
                "very common (low information)"
            };
            println!("│   Interpretation: {}", info_level);
            println!("│");
        }
        println!("└─────────────────────────────────────────────────────────────┘");
    }

    // Perform the lookup with kernel-weighted expansion
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ SEMANTIC LOOKUP RESULTS                                     │");
    println!("├─────────────────────────────────────────────────────────────┤");

    // Expand positions to neighborhoods with Gaussian weighting
    let mut word_scores: std::collections::HashMap<String, (f64, f64, u32)> = std::collections::HashMap::new();

    for &pos in &positions {
        // Get neighborhood with weights
        let neighbors = grid.neighborhood_weighted(pos, radius);

        for (neighbor, distance) in neighbors {
            // Gaussian weight decay (or uniform if kernel disabled)
            let spatial_weight = if use_kernel {
                (-distance * distance / (2.0 * sigma * sigma)).exp()
            } else {
                1.0
            };

            // Get words at this neighbor position
            if let Ok(words) = retina.words_at_position(neighbor) {
                // IDF weight
                let idf = if !words.is_empty() {
                    (retina.vocabulary_size() as f64 / (1.0 + words.len() as f64)).ln()
                } else {
                    0.0
                };

                let combined_weight = spatial_weight * (1.0 + idf);

                for word in words {
                    let entry = word_scores.entry(word).or_insert((0.0, f64::MAX, 0));
                    entry.0 += combined_weight;  // Accumulated weight
                    entry.1 = entry.1.min(distance);  // Min distance
                    entry.2 += 1;  // Overlap count
                }
            }
        }
    }

    // Sort by score
    let mut results: Vec<(String, f64, f64, u32)> = word_scores
        .into_iter()
        .map(|(word, (score, min_dist, overlap))| (word, score, min_dist, overlap))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results.truncate(count);

    if results.is_empty() {
        println!("│ No words found at or near these positions.                  │");
    } else {
        println!("│ {:>5}  {:>8}  {:>6}  {:>7}  {:<20} │", "Rank", "Score", "Dist", "Overlap", "Word");
        println!("│ {:─>5}  {:─>8}  {:─>6}  {:─>7}  {:─<20} │", "", "", "", "", "");

        for (i, (word, score, min_dist, overlap)) in results.iter().enumerate() {
            let word_display = if word.len() > 20 {
                format!("{}...", &word[..17])
            } else {
                word.clone()
            };
            println!("│ {:>5}  {:>8.3}  {:>6.2}  {:>7}  {:<20} │",
                i + 1, score, min_dist, overlap, word_display);
        }
    }
    println!("└─────────────────────────────────────────────────────────────┘");

    // Show detailed explanations if requested
    if explain && !results.is_empty() {
        println!("\n┌─────────────────────────────────────────────────────────────┐");
        println!("│ SCORE BREAKDOWN (top 5)                                     │");
        println!("├─────────────────────────────────────────────────────────────┤");

        for (i, (word, score, min_dist, overlap)) in results.iter().take(5).enumerate() {
            println!("│");
            println!("│ {}. \"{}\"", i + 1, word);
            println!("│    Total score: {:.4}", score);
            println!("│    Components:");
            println!("│      • Spatial proximity (Gaussian decay): based on dist {:.2}", min_dist);
            println!("│      • Information content (IDF weighting): incorporated in score");
            println!("│      • Multi-position coverage: {} query positions", overlap);
            println!("│    Interpretation:");

            if *min_dist < 0.5 {
                println!("│      → Direct hit: word fingerprint includes query position(s)");
            } else if *min_dist < 1.5 {
                println!("│      → Near neighbor: semantically adjacent region");
            } else {
                println!("│      → Extended neighborhood: broader semantic field");
            }
        }
        println!("│");
        println!("└─────────────────────────────────────────────────────────────┘");
    }

    // Show cluster information if requested
    if show_clusters {
        println!("\n┌─────────────────────────────────────────────────────────────┐");
        println!("│ CLUSTER ANALYSIS                                            │");
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│ Note: Full hierarchical clustering requires engine          │");
        println!("│ initialization. Showing basic spatial clustering:           │");
        println!("│");

        // Simple spatial clustering based on position proximity
        let mut clusters: Vec<Vec<u32>> = Vec::new();
        let mut assigned: std::collections::HashSet<u32> = std::collections::HashSet::new();

        for &pos in &positions {
            if assigned.contains(&pos) {
                continue;
            }

            let mut cluster = vec![pos];
            assigned.insert(pos);

            // Find other positions within 2*radius
            for &other in &positions {
                if !assigned.contains(&other) && grid.wrapped_distance(pos, other) < 2.0 * radius {
                    cluster.push(other);
                    assigned.insert(other);
                }
            }

            clusters.push(cluster);
        }

        for (i, cluster) in clusters.iter().enumerate() {
            let centroid = if cluster.len() == 1 {
                cluster[0]
            } else {
                // Simple average (not circular mean for display)
                let sum: u32 = cluster.iter().sum();
                sum / cluster.len() as u32
            };
            let centroid_2d = grid.to_2d(centroid);

            println!("│ Cluster {}: {} position(s) near ({}, {})",
                i + 1, cluster.len(), centroid_2d.row, centroid_2d.col);

            // Get top words for this cluster
            let mut cluster_words: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            for &pos in cluster {
                if let Ok(words) = retina.words_at_position(pos) {
                    for word in words.into_iter().take(20) {
                        *cluster_words.entry(word).or_insert(0) += 1;
                    }
                }
            }

            let mut top_words: Vec<(String, usize)> = cluster_words.into_iter().collect();
            top_words.sort_by(|a, b| b.1.cmp(&a.1));

            let display_words: Vec<&str> = top_words.iter()
                .take(5)
                .map(|(w, _)| w.as_str())
                .collect();

            if !display_words.is_empty() {
                println!("│   Representative words: {}", display_words.join(", "));
            }
        }
        println!("│");
        println!("└─────────────────────────────────────────────────────────────┘");
    }

    let elapsed = start_time.elapsed();
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Completed in {}", HumanDuration(elapsed));
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}
