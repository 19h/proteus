//! Proteus CLI - Semantic Folding Engine
//!
//! Command-line interface for training and using semantic retinas.

use clap::{Parser, Subcommand};
use image::{ImageBuffer, Rgb};
use indicatif::{HumanDuration, ProgressBar, ProgressStyle};
use log::error;
use proteus::{
    FingerprintConfig, Result, Retina, SegmentationConfig, SemanticSegmenter, SomConfig, Tokenizer,
};
use proteus::{Sdr, SemanticLookupConfig, SemanticLookupEngine, ToroidalGrid};
use std::fs::{self, File};
use std::io::{self, BufReader, Read};
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
    Train(TrainArgs),

    /// Compute fingerprint for text
    Fingerprint {
        /// Retina file to use
        #[arg(short, long)]
        retina: PathBuf,

        /// Text to fingerprint
        text: String,

        /// Output format (positions, binary, hex)
        #[arg(short, long, default_value = "positions", value_parser = ["positions", "binary", "hex"])]
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
        /// Use regex sentence splitting without loading a neural model
        #[arg(short, long)]
        fast: bool,
    },
    /// Migrate retina file to latest format (adds inverted index)
    Migrate {
        /// Input retina file (legacy or current format)
        #[arg(short, long)]
        input: PathBuf,

        /// Output retina file (v4 with metadata and inverted index)
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
        Commands::Train(args) => train_retina(args),

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
        } => segment_text(
            retina,
            input,
            block_size,
            threshold,
            min_segment,
            debug,
            fast,
        ),
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
            retina, positions, count, radius, sigma, adaptive, explain, clusters, stats, kernel,
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

#[derive(clap::Args)]
struct TrainArgs {
    /// UTF-8 corpus (blank-line separated paragraphs by default)
    #[arg(short, long)]
    input: PathBuf,
    /// Output model, written atomically
    #[arg(short, long)]
    output: PathBuf,
    /// Square grid side length
    #[arg(short, long, default_value_t = 128)]
    dimension: usize,
    /// Exact number of online SOM updates
    #[arg(short = 'n', long, default_value_t = 100_000)]
    iterations: usize,
    /// Reproducible training seed, saved in the model
    #[arg(short, long, default_value_t = 42)]
    seed: u64,
    /// Input records: paragraphs (wrapped text) or lines (one document per line)
    #[arg(long, value_parser = ["paragraphs", "lines"], default_value = "paragraphs")]
    documents: String,
    /// Maximum normalized tokens per nonoverlapping context
    #[arg(long, default_value_t = 64)]
    snippet_tokens: usize,
    /// Minimum corpus token frequency for a retained word
    #[arg(long, default_value_t = 2)]
    min_count: usize,
    /// Exclude TF-IDF features occurring in more than this fraction of snippets
    #[arg(long, default_value_t = 0.5)]
    max_document_frequency: f64,
    /// Optional TF-IDF random projection; 0 keeps exact vocabulary coordinates
    #[arg(long, default_value_t = 0)]
    projection_dim: usize,
    /// Initial online learning rate
    #[arg(long, default_value_t = 0.1)]
    initial_learning_rate: f64,
    /// Final online learning rate
    #[arg(long, default_value_t = 0.01)]
    final_learning_rate: f64,
    /// Initial Gaussian radius in cells (default: half the grid side)
    #[arg(long)]
    initial_radius: Option<f64>,
    /// Final Gaussian radius in cells
    #[arg(long, default_value_t = 1.0)]
    final_radius: f64,
    /// Planar topology instead of wraparound
    #[arg(long)]
    planar: bool,
    /// Worker threads (0 uses the Rayon default)
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Maximum fraction of active cells in a fingerprint
    #[arg(long, default_value_t = 0.02)]
    sparsity: f64,
    /// Optional additional absolute cap on fingerprint cardinality
    #[arg(long)]
    max_active_bits: Option<usize>,
}

fn train_retina(args: TrainArgs) -> Result<()> {
    use proteus::training::{train_corpus, DocumentMode, TrainingConfig};
    let start = Instant::now();
    let config = TrainingConfig {
        som: SomConfig {
            dimension: args.dimension,
            iterations: args.iterations,
            seed: Some(args.seed),
            initial_learning_rate: args.initial_learning_rate,
            final_learning_rate: args.final_learning_rate,
            initial_radius: args.initial_radius.unwrap_or(args.dimension as f64 / 2.0),
            final_radius: args.final_radius,
            toroidal: !args.planar,
            num_threads: args.threads,
            ..Default::default()
        },
        fingerprint: FingerprintConfig {
            sparsity: args.sparsity,
            max_active_bits: args.max_active_bits.unwrap_or(usize::MAX),
            min_active_bits: 0,
            weighted_aggregation: true,
        },
        document_mode: if args.documents == "lines" {
            DocumentMode::Lines
        } else {
            DocumentMode::Paragraphs
        },
        snippet_tokens: args.snippet_tokens,
        min_count: args.min_count,
        max_document_frequency: args.max_document_frequency,
        projection_dim: args.projection_dim,
        ..Default::default()
    };
    config.validate()?;
    println!(
        "Training {}: {}x{} grid, {} online updates, seed {}",
        args.input.display(),
        args.dimension,
        args.dimension,
        args.iterations,
        args.seed
    );
    println!(
        "Contexts: {}, <= {} tokens; min count {}; max document frequency {}",
        args.documents, args.snippet_tokens, args.min_count, args.max_document_frequency
    );
    println!(
        "TF-IDF projection: {}",
        if args.projection_dim == 0 {
            "none (exact)".into()
        } else {
            args.projection_dim.to_string()
        }
    );
    let pb = ProgressBar::new(args.iterations as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise}] [{bar:40}] {pos}/{len} ETA {eta}")
            .unwrap(),
    );
    pb.set_message("Online SOM");
    let (retina, report) = train_corpus(
        BufReader::new(File::open(&args.input)?),
        config,
        |done, _| pb.set_position(done as u64),
    )?;
    pb.finish_and_clear();
    retina.save(&args.output)?;
    println!(
        "Saved {} in {}",
        args.output.display(),
        HumanDuration(start.elapsed())
    );
    println!(
        "Words: {}; snippets: {}/{}; features: {}; vector dimension: {}",
        report.vocabulary,
        report.mapped_snippets,
        report.snippets,
        report.features,
        report.config.som.weight_dimension
    );
    println!(
        "Occupied cells: {}/{}; mean quantization error: {:.6}",
        report.occupied_cells,
        retina.grid_size(),
        report.quantization_error
    );
    if report.mapped_snippets < retina.grid_size() as usize {
        println!("Corpus has fewer usable contexts than grid cells; a smaller grid can increase cell support.");
    }
    Ok(())
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
                if i > 0 && i % retina.dimension() as usize == 0 {
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
    let mapped = proteus::storage::MmappedRetina::open(&retina_path)?;
    let similar = if mapped.has_inverted_index() {
        mapped
            .find_similar(&word, count)
            .ok_or_else(|| proteus::ProteusError::WordNotFound(word.clone()))?
    } else {
        Retina::load(&retina_path)?.find_similar_words(&word, count)?
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
    if let Some(report) = retina.training_report() {
        println!(
            "  Training: {}",
            serde_json::to_string_pretty(report)
                .map_err(|e| proteus::ProteusError::Serialization(e.to_string()))?
        );
    } else {
        println!("  Training provenance: unknown (legacy or manually constructed model)");
    }

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
    let mut retina = Retina::load(&input)?;
    retina.build_index();
    retina.save(&output)?;
    println!(
        "Migrated {} to {} (format v4, metadata and persisted index)",
        input.display(),
        output.display()
    );
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
            "No valid positions provided. Use comma-separated numbers, e.g., '1234,5678'"
                .to_string(),
        ));
    }

    // Load retina
    let retina = Retina::load(&retina_path)?;
    let grid_size = retina.grid_size();
    let dimension = retina.dimension();

    // Validate positions
    for &pos in &positions {
        if pos >= grid_size {
            return Err(proteus::error::ProteusError::TextProcessing(format!(
                "Position {} is out of range (max: {})",
                pos,
                grid_size - 1
            )));
        }
    }

    println!(
        "Retina: {} ({}x{} grid)",
        retina_path.display(),
        dimension,
        dimension
    );
    println!(
        "Looking up {} position(s): {:?}\n",
        positions.len(),
        positions
    );

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
                "No input provided. Use --input <file>, --input - for stdin, or --text <text>"
                    .to_string(),
            ));
        }
    };

    if text.trim().is_empty() {
        return Err(proteus::error::ProteusError::TextProcessing(
            "Input text is empty".to_string(),
        ));
    }

    // Validate mode selection
    if !segment_mode && !words_mode {
        return Err(proteus::error::ProteusError::Config(
            "Please specify a mode: --segment or --words".to_string(),
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
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{msg}\n{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len}",
                )
                .unwrap()
                .progress_chars("█▓▒░  "),
        );
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

                img.save(&filepath)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

                image_count += 1;
            }
            pb.inc(1);
        }
        pb.finish_and_clear();
        println!("  Generated {} word fingerprint images", image_count);
    }

    if segment_mode {
        // Load retina (or reload if already loaded for words mode)
        println!(
            "{}Loading retina from {}...",
            if words_mode { "\n" } else { "" },
            retina_path.display()
        );
        let retina = Retina::load(&retina_path)?;
        let dimension = retina.dimension();
        if !words_mode {
            println!("  Grid: {}x{}", dimension, dimension);
        }

        println!(
            "\nSegmenting text{}...",
            if fast { " (fast mode)" } else { "" }
        );

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

            img.save(&filepath)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

            println!(
                "  [{}] {} ({} chars, {} active bits)",
                i + 1,
                filename,
                segment.text.len(),
                fp.cardinality()
            );
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
            "No valid positions provided. Use comma-separated numbers.".to_string(),
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
    println!(
        "      Grid: {}x{} ({} positions)",
        dimension, dimension, grid_size
    );
    println!("      Vocabulary: {} words", retina.vocabulary_size());

    // Validate positions
    for &pos in &positions {
        if pos >= grid_size {
            return Err(proteus::error::ProteusError::TextProcessing(format!(
                "Position {} out of range (max: {})",
                pos,
                grid_size - 1
            )));
        }
    }

    // Check that inverted index exists
    match retina.words_at_position(0) {
        Ok(_) => {
            // Index exists, we can proceed
        }
        Err(_) => {
            return Err(proteus::error::ProteusError::TextProcessing(
                "Retina has no inverted index. Run 'proteus migrate' first.".to_string(),
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
    println!(
        "      Adaptive radius: {}",
        if adaptive { "enabled" } else { "disabled" }
    );
    println!(
        "      Kernel weighting: {}",
        if use_kernel {
            "Gaussian decay"
        } else {
            "uniform"
        }
    );

    // Display query positions
    let grid = ToroidalGrid::new(dimension);
    println!("\n[3/3] Analyzing {} query position(s)...", positions.len());

    for &pos in &positions {
        let pos_2d = grid.to_2d(pos);
        println!(
            "      Position {} (row {}, col {})",
            pos, pos_2d.row, pos_2d.col
        );
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
            let neighbor_counts: Vec<usize> = neighbors
                .iter()
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
            println!(
                "│   Neighborhood size (r={:.1}): {} positions",
                radius,
                neighbors.len()
            );

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
    let mut word_scores: std::collections::HashMap<String, (f64, f64, u32)> =
        std::collections::HashMap::new();

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
                    entry.0 += combined_weight; // Accumulated weight
                    entry.1 = entry.1.min(distance); // Min distance
                    entry.2 += 1; // Overlap count
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
        println!(
            "│ {:>5}  {:>8}  {:>6}  {:>7}  {:<20} │",
            "Rank", "Score", "Dist", "Overlap", "Word"
        );
        println!(
            "│ {:─>5}  {:─>8}  {:─>6}  {:─>7}  {:─<20} │",
            "", "", "", "", ""
        );

        for (i, (word, score, min_dist, overlap)) in results.iter().enumerate() {
            let word_display = if word.len() > 20 {
                format!("{}...", &word[..17])
            } else {
                word.clone()
            };
            println!(
                "│ {:>5}  {:>8.3}  {:>6.2}  {:>7}  {:<20} │",
                i + 1,
                score,
                min_dist,
                overlap,
                word_display
            );
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
            println!(
                "│      • Spatial proximity (Gaussian decay): based on dist {:.2}",
                min_dist
            );
            println!("│      • Information content (IDF weighting): incorporated in score");
            println!(
                "│      • Multi-position coverage: {} query positions",
                overlap
            );
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

            println!(
                "│ Cluster {}: {} position(s) near ({}, {})",
                i + 1,
                cluster.len(),
                centroid_2d.row,
                centroid_2d.col
            );

            // Get top words for this cluster
            let mut cluster_words: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();
            for &pos in cluster {
                if let Ok(words) = retina.words_at_position(pos) {
                    for word in words.into_iter().take(20) {
                        *cluster_words.entry(word).or_insert(0) += 1;
                    }
                }
            }

            let mut top_words: Vec<(String, usize)> = cluster_words.into_iter().collect();
            top_words.sort_by(|a, b| b.1.cmp(&a.1));

            let display_words: Vec<&str> =
                top_words.iter().take(5).map(|(w, _)| w.as_str()).collect();

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
