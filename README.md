# Proteus

**A high-performance Semantic Folding engine in Rust**

Proteus implements [Semantic Folding](https://www.cortical.io/science/), a biologically-inspired approach to natural language processing that represents text as Sparse Distributed Representations (SDRs). Unlike traditional word embeddings, SDRs encode meaning in the pattern of active bits, enabling efficient similarity computation through simple bitwise operations.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Usage](#cli-usage)
- [Library Usage](#library-usage)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Performance](#performance)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Scaling to Terabytes & Cross-Lingual Alignment](#scaling-to-terabytes--cross-lingual-alignment)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

Semantic Folding transforms natural language into sparse binary fingerprints on a 2D semantic space. Each position in this space represents a learned semantic context, and words are represented by the set of contexts they appear in. This representation has several compelling properties:

- **Semantic similarity through overlap**: Similar concepts share active bits
- **Compositionality**: Text fingerprints are unions of word fingerprints
- **Interpretability**: Each bit position corresponds to a learned semantic context
- **Efficiency**: Similarity computation reduces to fast bitwise AND operations
- **Robustness**: SDRs are inherently noise-tolerant due to distributed encoding

Proteus is a complete, from-scratch implementation in Rust - not a port of existing Python/Java implementations. It achieves high performance through SIMD operations, parallel processing, and memory-efficient data structures.

## Features

### Core Capabilities

- **Self-Organizing Map (SOM)**: Neural network that learns semantic topology from text corpora
- **Sparse Distributed Representations**: Efficient binary fingerprints using RoaringBitmaps
- **Multiple Similarity Measures**: Cosine, Jaccard, and Overlap (Szymkiewicz-Simpson) coefficients
- **Fast Similarity Search**: Inverted index for approximate nearest neighbor queries
- **Semantic Operations**: Analogy (A:B::C:?), semantic center, semantic difference

### Text Processing

- **Unicode-aware tokenization**: Proper handling of international text via `unicode-segmentation`
- **Configurable normalization**: Lowercase, punctuation removal, NFD normalization
- **Sentence segmentation**: Neural sentence boundary detection using [wtpsplit](https://github.com/bminixhofer/wtpsplit) (SaT models)
- **Semantic segmentation**: Topic-based text segmentation using TextTiling algorithm with SDR fingerprints
- **Context window extraction**: Sentence-aware windows that don't cross boundaries

### Storage & Persistence

- **Efficient binary format**: Custom format optimized for random access
- **Memory-mapped files**: Handle large retinas without loading into memory
- **Serialization**: Full serde support for all data structures

### Performance

- **Parallel training**: Multi-threaded SOM training via Rayon
- **SIMD optimization**: Vectorized distance calculations (f32)
- **Compressed bitmaps**: RoaringBitmap for space-efficient SDR storage
- **Optimized search**: Inverted index with RoaringBitmap posting lists

### Advanced Indexing & Search

- **HNSW Index**: Hierarchical Navigable Small World graphs for O(log n) approximate nearest neighbor search
- **Product Quantization**: ~100x compression with asymmetric distance computation
- **Locality-Sensitive Hashing**: SimHash, MinHash, and BitSampling for sublinear similarity search
- **Mini-batch SOM**: Modern training with Adam optimizer and cosine annealing
- **Hyperdimensional Computing**: Vector Symbolic Architectures for symbolic reasoning on SDRs

## Installation

### Prerequisites

- Rust 1.70 or later
- ONNX Runtime (for sentence segmentation features)

### Building from Source

```bash
git clone https://github.com/proteus/proteus.git
cd proteus
cargo build --release
```

The binary will be available at `target/release/proteus`.

### As a Library Dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
proteus = { git = "https://github.com/proteus/proteus.git" }
```

## Quick Start

### Training a Retina

A "retina" is Proteus's name for a trained semantic model. Train one from a text corpus:

```bash
# Train on a corpus (one document per line)
proteus train -i corpus.txt -o english.retina -d 128 -n 100000

# With verbose output
proteus -v train -i corpus.txt -o english.retina
```

### Using a Trained Retina

```bash
# Generate a fingerprint for text
proteus fingerprint -r english.retina "artificial intelligence"

# Find similar words
proteus similar -r english.retina "king" -k 10

# Compute text similarity
proteus similarity -r english.retina "machine learning" "deep learning"

# View retina statistics
proteus info english.retina
```

## CLI Usage

### Commands

#### `proteus train`

Train a new retina from a text corpus.

```
USAGE:
    proteus train [OPTIONS] -i <INPUT> -o <OUTPUT>

OPTIONS:
    -i, --input <INPUT>        Input corpus file (one document per line)
    -o, --output <OUTPUT>      Output retina file
    -d, --dimension <DIM>      Grid dimension [default: 128]
    -n, --iterations <N>       Training iterations [default: 100000]
    -s, --seed <SEED>          Random seed for reproducibility
    -v, --verbose              Enable verbose logging
```

The grid dimension determines the semantic space size: a 128x128 grid has 16,384 positions, with ~2% (328 bits) active per fingerprint.

#### `proteus fingerprint`

Generate a fingerprint for text input.

```
USAGE:
    proteus fingerprint -r <RETINA> <TEXT> [--format <FORMAT>]

OPTIONS:
    -r, --retina <RETINA>      Retina file to use
    -f, --format <FORMAT>      Output format: positions, binary, hex [default: positions]
```

Output formats:
- `positions`: List of active bit positions `[45, 892, 1203, ...]`
- `binary`: 128x128 grid of 0s and 1s
- `hex`: Compact hexadecimal encoding

#### `proteus similar`

Find words semantically similar to a query word.

```
USAGE:
    proteus similar -r <RETINA> <WORD> [-k <COUNT>]

OPTIONS:
    -r, --retina <RETINA>      Retina file to use
    -k, --count <COUNT>        Number of results [default: 10]
```

#### `proteus similarity`

Compute semantic similarity between two texts.

```
USAGE:
    proteus similarity -r <RETINA> <TEXT1> <TEXT2>
```

Returns a value between 0.0 (unrelated) and 1.0 (identical).

#### `proteus info`

Display retina statistics.

```
USAGE:
    proteus info <RETINA>
```

Shows grid dimension, total positions, and vocabulary size.

#### `proteus segment`

Segment text into semantic sections based on topic shifts.

```
USAGE:
    proteus segment [OPTIONS] -r <RETINA> -i <INPUT>

OPTIONS:
    -r, --retina <RETINA>            Retina file to use
    -i, --input <INPUT>              Input text file
    -b, --block-size <BLOCK_SIZE>    Sentences per block for comparison [default: 3]
    -t, --threshold <THRESHOLD>      Depth threshold in std deviations [default: 0.5]
    -m, --min-segment <MIN_SEGMENT>  Minimum sentences per segment [default: 2]
        --debug                      Show similarity/depth scores for analysis
```

Uses the TextTiling algorithm (Hearst, 1997) adapted for SDR fingerprints to detect topic boundaries. The algorithm:
1. Segments text into sentences using neural sentence boundary detection
2. Groups sentences into blocks and computes block fingerprints
3. Measures similarity between adjacent blocks
4. Identifies boundaries where similarity drops significantly (depth score analysis)

Example:
```bash
# Basic segmentation
proteus segment -r english.retina -i article.txt

# With debug output showing similarity analysis
proteus segment -r english.retina -i article.txt --debug

# Adjust sensitivity (higher threshold = fewer boundaries)
proteus segment -r english.retina -i article.txt --threshold 1.0
```

## Library Usage

### Basic Operations

```rust
use proteus::{Retina, Config, Result};

fn main() -> Result<()> {
    // Load a pre-trained retina
    let retina = Retina::load("english.retina")?;

    // Generate text fingerprint
    let fp = retina.fingerprint_text("The quick brown fox");
    println!("Active bits: {}", fp.cardinality());
    println!("Sparsity: {:.2}%", fp.sparsity() * 100.0);

    // Compute text similarity
    let sim = retina.text_similarity(
        "machine learning algorithms",
        "deep neural networks"
    );
    println!("Similarity: {:.4}", sim);

    // Find similar words
    let similar = retina.find_similar_words("king", 5)?;
    for (word, score) in similar {
        println!("  {:.4}  {}", score, word);
    }

    Ok(())
}
```

### Training a New Retina

```rust
use proteus::{
    Config, Som, SomConfig, SomTrainer,
    WordFingerprinter, FingerprintConfig, Retina, Tokenizer,
    Result
};
use proteus::som::training::{WordEmbeddings, DEFAULT_EMBEDDING_DIM};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn train_retina(corpus_path: &str, output_path: &str) -> Result<()> {
    // Read corpus (one document per line)
    let file = File::open(corpus_path)?;
    let documents: Vec<String> = BufReader::new(file)
        .lines()
        .filter_map(|l| l.ok())
        .collect();

    // Tokenize and extract context windows
    let tokenizer = Tokenizer::default_config();
    let mut contexts: Vec<(String, Vec<String>)> = Vec::new();

    for doc in &documents {
        for (center, context) in tokenizer.context_windows_as_strings(doc, 2) {
            contexts.push((center, context));
        }
    }

    // Learn word embeddings from context co-occurrence
    let embeddings = WordEmbeddings::from_contexts(
        &contexts,
        DEFAULT_EMBEDDING_DIM,
        Some(42)  // Random seed
    );

    // Configure and train SOM
    let som_config = SomConfig {
        dimension: 128,
        weight_dimension: DEFAULT_EMBEDDING_DIM,
        iterations: 100_000,
        seed: Some(42),
        ..Default::default()
    };

    let mut som = Som::new(&som_config);
    let mut trainer = SomTrainer::new(som_config);
    let word_to_bmus = trainer.train_fast(&mut som, &embeddings, &contexts)?;

    // Generate fingerprints from BMU mappings
    let grid_size = 128 * 128;
    let fp_config = FingerprintConfig::default();
    let mut fingerprinter = WordFingerprinter::new(fp_config.clone(), grid_size);
    fingerprinter.create_fingerprints(&word_to_bmus, None);

    // Create and save retina
    let retina = Retina::with_index(
        fingerprinter.into_fingerprints(),
        128,
        fp_config
    );
    retina.save(output_path)?;

    Ok(())
}
```

### Advanced Semantic Operations

```rust
use proteus::{Retina, Result};

fn semantic_operations(retina: &Retina) -> Result<()> {
    // Semantic center: find common ground between concepts
    let center = retina.semantic_center(&["king", "queen", "prince", "princess"]);
    let related = retina.find_similar_to_fingerprint(&center, 5);
    println!("Semantic center of royalty: {:?}", related);

    // Semantic difference: what makes concept A different from B?
    if let Some(diff) = retina.semantic_difference("king", "queen") {
        let unique_to_king = retina.find_similar_to_fingerprint(&diff, 5);
        println!("What 'king' has that 'queen' doesn't: {:?}", unique_to_king);
    }

    // Analogy: king is to queen as man is to ???
    let results = retina.analogy("king", "queen", "man", 5)?;
    println!("king:queen :: man:?");
    for (word, score) in results {
        println!("  {:.4}  {}", score, word);
    }

    Ok(())
}
```

### Sentence-Aware Processing

```rust
use proteus::{Tokenizer, SentenceSegmenter, Result};

fn sentence_aware_tokenization() -> Result<()> {
    let text = "Dr. Smith went to Washington. He met with Sen. Johnson.";

    // Create sentence segmenter (downloads model on first use)
    let mut segmenter = SentenceSegmenter::new("sat-3l-sm")?;

    // Segment into sentences
    let sentences = segmenter.segment(text)?;
    for sentence in &sentences {
        println!("Sentence: {}", sentence);
    }

    // Create context windows that don't cross sentence boundaries
    let tokenizer = Tokenizer::default_config();
    let windows = tokenizer.sentence_aware_context_windows(
        text,
        2,  // half_window: 2 tokens on each side
        &mut segmenter
    )?;

    for (center, context) in windows {
        println!("{}: {:?}", center, context);
    }

    Ok(())
}
```

### Semantic Text Segmentation

```rust
use proteus::{Retina, SemanticSegmenter, SegmentationConfig, Result};

fn segment_into_topics() -> Result<()> {
    let retina = Retina::load("english.retina")?;

    // Create segmenter with custom configuration
    let config = SegmentationConfig {
        block_size: 3,              // Sentences per block
        depth_threshold: 0.5,       // Std deviations above mean for boundaries
        min_segment_sentences: 2,   // Minimum sentences per segment
        smoothing: true,            // Apply Gaussian smoothing
        smoothing_sigma: 1.0,
    };

    let mut segmenter = SemanticSegmenter::with_config(retina, config)?;

    let text = std::fs::read_to_string("article.txt")?;
    let result = segmenter.segment(&text)?;

    // Access detailed analysis
    println!("Found {} segments with {} boundaries",
        result.segments.len(),
        result.boundary_indices.len());
    println!("Depth scores: mean={:.3}, std={:.3}",
        result.depth_mean, result.depth_std);

    // Output segments
    for (i, segment) in result.segments.iter().enumerate() {
        println!("\n=== Segment {} ({} sentences) ===",
            i + 1, segment.sentences.len());
        println!("{}", segment.text);
    }

    Ok(())
}
```

### Working with SDRs Directly

```rust
use proteus::Sdr;

fn sdr_operations() {
    let size = 16384;  // 128x128 grid

    // Create SDRs from active positions
    let sdr_a = Sdr::from_positions(&[100, 200, 300, 400, 500], size);
    let sdr_b = Sdr::from_positions(&[100, 200, 600, 700, 800], size);

    // Basic properties
    println!("Cardinality A: {}", sdr_a.cardinality());  // 5
    println!("Sparsity A: {:.4}", sdr_a.sparsity());     // 0.0003

    // Set operations
    let overlap = &sdr_a & &sdr_b;   // Positions in both
    let union = &sdr_a | &sdr_b;     // Positions in either
    let diff = &sdr_a ^ &sdr_b;      // Positions in one but not both

    println!("Overlap count: {}", overlap.cardinality());  // 2

    // Similarity measures
    println!("Jaccard: {:.4}", sdr_a.jaccard_similarity(&sdr_b));
    println!("Cosine: {:.4}", sdr_a.cosine_similarity(&sdr_b));
    println!("Overlap: {:.4}", sdr_a.overlap_similarity(&sdr_b));

    // Convert to dense representation
    let dense = sdr_a.to_dense();  // Vec<u8> of 0s and 1s

    // Sparsify (limit active bits)
    let mut sdr = Sdr::from_positions(&(0..1000).collect::<Vec<_>>(), size);
    sdr.sparsify(328);  // Keep at most 328 active bits
}
```

### Advanced Indexing Algorithms

Proteus includes state-of-the-art indexing algorithms for large-scale similarity search:

#### HNSW (Hierarchical Navigable Small World)

O(log n) approximate nearest neighbor search using multi-layer navigable graphs:

```rust
use proteus::index::{HnswIndex, HnswConfig};
use proteus::Sdr;

fn hnsw_search() {
    // Configure HNSW parameters
    let config = HnswConfig {
        m: 32,                  // Max connections per node at layer 0
        m_max: 16,              // Max connections at higher layers
        ef_construction: 200,   // Build-time candidate list size
        ef_search: 50,          // Query-time candidate list size
        ml: 1.0 / (32_f64).ln(), // Level generation factor
        seed: Some(42),
    };

    // Build index from fingerprints
    let fingerprints: Vec<Sdr> = load_fingerprints();
    let index = HnswIndex::build(config, fingerprints);

    // Search for k nearest neighbors
    let query = Sdr::from_positions(&[100, 200, 300], 16384);
    let results = index.search(&query, 10);  // Returns Vec<(id, distance)>

    for (id, dist) in results {
        println!("ID: {}, Distance: {:.4}", id, dist);
    }

    // Get index statistics
    let stats = index.stats();
    println!("Nodes: {}, Max layer: {}", stats.num_nodes, stats.max_layer);
}
```

#### Product Quantization (PQ)

Compress vectors for memory-efficient storage with fast distance computation:

```rust
use proteus::index::{ProductQuantizer, PqConfig};

fn product_quantization() {
    let config = PqConfig {
        num_subquantizers: 8,   // M partitions (must divide vector dim)
        num_centroids: 256,     // K centroids per subquantizer
        kmeans_iterations: 25,
        seed: Some(42),
    };

    // Train on representative vectors
    let training_vectors: Vec<Vec<f32>> = load_training_data();
    let pq = ProductQuantizer::train(&training_vectors, config);

    // Encode vectors (compression)
    let original = vec![0.1, 0.2, 0.3, /* ... */];
    let codes = pq.encode(&original);  // Vec<u8> - much smaller!

    // Compute distances without full decompression (ADC)
    let query = vec![0.15, 0.25, 0.35, /* ... */];
    let distance = pq.asymmetric_distance(&query, &codes);

    // Decode back to approximate vector
    let reconstructed = pq.decode(&codes);
}
```

#### Locality-Sensitive Hashing (LSH)

Sublinear similarity search using hash-based bucketing:

```rust
use proteus::index::{SimHash, MinHash, BitSamplingLsh, LshIndex};
use proteus::Sdr;

fn lsh_search() {
    // SimHash for cosine similarity on dense vectors
    let simhash = SimHash::new(128, 64, Some(42));  // dim, num_bits, seed
    let hash1 = simhash.hash(&dense_vector1);
    let hash2 = simhash.hash(&dense_vector2);
    let hamming_dist = (hash1 ^ hash2).count_ones();

    // MinHash for Jaccard similarity on sets
    let minhash = MinHash::new(128, Some(42));  // num_hashes, seed
    let sig1 = minhash.signature(&set1);
    let sig2 = minhash.signature(&set2);
    let approx_jaccard = MinHash::estimate_jaccard(&sig1, &sig2);

    // BitSampling for sparse binary vectors (SDRs)
    let bitsampling = BitSamplingLsh::new(16384, 64, Some(42));
    let sdr_hash = bitsampling.hash(&sdr);

    // Multi-table LSH index for fast retrieval
    let sdrs: Vec<Sdr> = load_fingerprints();
    let mut index = LshIndex::new(bitsampling, 10, 4);  // hasher, tables, bands
    index.build(&sdrs);

    let query = Sdr::from_positions(&[100, 200, 300], 16384);
    let candidates = index.query(&query);  // Returns candidate IDs
}
```

#### Mini-batch SOM Training

Modern SOM training with adaptive optimization:

```rust
use proteus::som::{BatchSomTrainer, BatchSomConfig, Som, SomConfig};

fn minibatch_training() {
    let config = BatchSomConfig {
        batch_size: 256,
        epochs: 10,
        use_adam: true,           // Adam optimizer
        adam_beta1: 0.9,
        adam_beta2: 0.999,
        use_cosine_annealing: true,
        warm_restarts: 2,         // Restart count for SGDR
        initial_lr: 0.1,
        final_lr: 0.001,
        initial_radius: 32.0,
        final_radius: 1.0,
    };

    let som_config = SomConfig {
        dimension: 64,
        weight_dimension: 100,
        ..Default::default()
    };

    let mut som = Som::new(&som_config);
    let mut trainer = BatchSomTrainer::new(config);

    // Train with monitoring
    let training_data: Vec<Vec<f32>> = load_embeddings();
    let metrics = trainer.train(&mut som, &training_data);

    println!("Final QE: {:.4}", metrics.quantization_error);
    println!("Topographic error: {:.4}", metrics.topographic_error);
}
```

### Hyperdimensional Computing (HDC)

Vector Symbolic Architectures for symbolic reasoning on high-dimensional binary vectors:

```rust
use proteus::fingerprint::{Hypervector, ItemMemory, SequenceEncoder, AnalogySolver};

fn hyperdimensional_computing() {
    // Create random hypervectors (50% density for HDC operations)
    let hv_king = Hypervector::random(10000, 0.5, Some(1));
    let hv_queen = Hypervector::random(10000, 0.5, Some(2));
    let hv_man = Hypervector::random(10000, 0.5, Some(3));
    let hv_woman = Hypervector::random(10000, 0.5, Some(4));

    // Bundling (⊕): Combine vectors (soft union)
    let royalty = Hypervector::bundle(&[&hv_king, &hv_queen], 0.5);
    assert!(royalty.similarity(&hv_king) > 0.3);  // Similar to both

    // Binding (⊗): Create associations via XOR
    let king_male = hv_king.bind(&hv_man);
    let queen_female = hv_queen.bind(&hv_woman);

    // Unbinding recovers the other component
    let recovered = king_male.unbind(&hv_king);
    assert!(recovered.similarity(&hv_man) > 0.9);

    // Permutation (ρ): Position encoding for sequences
    let shifted = hv_king.permute(100);
    let restored = shifted.permute_inverse(100);
    assert_eq!(hv_king.similarity(&restored), 1.0);

    // Item Memory: Content-addressable storage
    let mut memory = ItemMemory::new(10000, 0.5);
    memory.encode("apple", Some(1));
    memory.encode("banana", Some(2));
    memory.encode("cherry", Some(3));

    let apple_hv = memory.get("apple").unwrap();
    let (name, similarity) = memory.query(apple_hv).unwrap();
    assert_eq!(name, "apple");

    // Sequence Encoding: Order-preserving representations
    let mut encoder = SequenceEncoder::new(10000, 0.5, 10);
    let seq1 = encoder.encode_sequence(&["the", "cat", "sat"]);
    let seq2 = encoder.encode_sequence(&["the", "dog", "sat"]);
    let seq3 = encoder.encode_sequence(&["a", "bird", "flew"]);

    // Similar sequences have higher similarity
    assert!(seq1.sdr.jaccard_similarity(&seq2.sdr) >
            seq1.sdr.jaccard_similarity(&seq3.sdr));

    // Analogy Solving: king - man + woman ≈ queen
    let mut solver = AnalogySolver::new(10000, 0.5);
    solver.add_concept("king", Some(1));
    solver.add_concept("queen", Some(2));
    solver.add_concept("man", Some(3));
    solver.add_concept("woman", Some(4));

    let results = solver.solve("king", "man", "woman", 3);
    // Returns concepts most similar to (king - man + woman)
}
```

### Custom Configuration

```rust
use proteus::{Config, SomConfig, TextConfig, FingerprintConfig};

fn custom_config() -> Config {
    Config {
        som: SomConfig {
            dimension: 64,              // Smaller grid (64x64 = 4,096 positions)
            iterations: 50_000,         // Fewer training iterations
            initial_learning_rate: 0.1,
            final_learning_rate: 0.01,
            initial_radius: 32.0,       // Half of dimension
            final_radius: 1.0,
            weight_dimension: 100,      // Compact embeddings
            context_window: 5,
            toroidal: true,             // Wrap-around boundaries
            seed: Some(42),             // Reproducibility
            num_threads: 0,             // Use all cores
        },
        text: TextConfig {
            lowercase: true,
            min_token_length: 2,
            max_token_length: 50,
            remove_punctuation: true,
            remove_numbers: false,
            unicode_normalize: true,
        },
        fingerprint: FingerprintConfig {
            sparsity: 0.02,             // 2% active bits
            max_active_bits: 82,        // 2% of 4,096
            min_active_bits: 5,
            weighted_aggregation: true,
        },
        ..Default::default()
    }
}
```

## Architecture

```
proteus/
├── src/
│   ├── lib.rs              # Library root with public API
│   ├── main.rs             # CLI entry point
│   ├── config.rs           # Configuration structs
│   ├── error.rs            # Error types (thiserror-based)
│   │
│   ├── text/               # Text Processing
│   │   ├── mod.rs
│   │   ├── tokenizer.rs    # Unicode-aware tokenization
│   │   ├── normalizer.rs   # Text normalization (lowercase, NFD)
│   │   └── segmenter.rs    # Sentence segmentation (wtpsplit)
│   │
│   ├── wtpsplit/           # Vendored wtpsplit-rs
│   │   ├── sat.rs          # SaT model (XLM-RoBERTa based)
│   │   ├── model.rs        # ONNX runtime wrapper
│   │   └── hub.rs          # HuggingFace Hub integration
│   │
│   ├── som/                # Self-Organizing Map
│   │   ├── mod.rs
│   │   ├── map.rs          # SOM grid implementation
│   │   ├── neuron.rs       # Neuron with weight vectors
│   │   ├── training.rs     # Training algorithms
│   │   ├── batch.rs        # Mini-batch training with Adam optimizer
│   │   └── simd.rs         # SIMD-accelerated operations
│   │
│   ├── fingerprint/        # Fingerprint Generation
│   │   ├── mod.rs
│   │   ├── sdr.rs          # SDR with RoaringBitmap
│   │   ├── word.rs         # Word fingerprinting
│   │   ├── text.rs         # Text fingerprinting
│   │   └── hdc.rs          # Hyperdimensional Computing operations
│   │
│   ├── segmentation/       # Semantic Segmentation
│   │   ├── mod.rs
│   │   └── semantic.rs     # TextTiling algorithm with SDRs
│   │
│   ├── roaring/            # Vendored RoaringBitmap
│   │
│   ├── storage/            # Persistence
│   │   ├── mod.rs
│   │   ├── format.rs       # Binary format (header + index + data)
│   │   └── retina.rs       # Retina management
│   │
│   ├── index/              # Search & Indexing
│   │   ├── mod.rs
│   │   ├── inverted.rs     # Inverted index for similarity search
│   │   ├── hnsw.rs         # HNSW graph index for ANN search
│   │   ├── pq.rs           # Product Quantization compression
│   │   └── lsh.rs          # Locality-Sensitive Hashing
│   │
│   └── similarity/         # Similarity Measures
│       ├── mod.rs
│       ├── cosine.rs
│       ├── jaccard.rs
│       └── overlap.rs
│
└── tests/
    └── integration_tests.rs
```

## How It Works

### The Semantic Folding Pipeline

```
                    ┌──────────────────────────────────────────────────┐
                    │                 TRAINING PHASE                   │
                    └──────────────────────────────────────────────────┘
                                          │
     ┌────────────┐      ┌────────────┐   │   ┌────────────┐      ┌────────────┐
     │   Corpus   │ ───▶ │  Tokenize  │ ──┼─▶ │  Context   │ ───▶ │   Word     │
     │   (text)   │      │ & Normalize│   │   │  Windows   │      │ Embeddings │
     └────────────┘      └────────────┘   │   └────────────┘      └────────────┘
                                          │                             │
                                          │                             ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │                                                          │
                    │                 Self-Organizing Map                      │
                    │                                                          │
                    │    ┌───┬───┬───┬───┬───┬───┬───┬───┐                     │
                    │    │   │   │   │   │   │   │   │   │  128x128 grid       │
                    │    ├───┼───┼───┼───┼───┼───┼───┼───┤  of neurons         │
                    │    │   │   │ ● │ ● │   │   │   │   │                     │
                    │    ├───┼───┼───┼───┼───┼───┼───┼───┤  Each neuron has    │
                    │    │   │   │ ● │ ● │ ● │   │   │   │  a weight vector    │
                    │    ├───┼───┼───┼───┼───┼───┼───┼───┤                     │
                    │    │   │   │   │ ● │   │   │   │   │  Similar contexts   │
                    │    └───┴───┴───┴───┴───┴───┴───┴───┘  cluster together   │
                    │                                                          │
                    └──────────────────────────────────────────────────────────┘
                                          │
                                          ▼
     ┌────────────┐      ┌────────────────────────────────────────────────────┐
     │   Word     │      │              Word Fingerprints                     │
     │  "king"    │ ───▶ │                                                    │
     └────────────┘      │    SDR: positions where word's contexts appear     │
                         │    [45, 892, 1203, 5670, 8921, 12456, ...]         │
                         │    ~328 active bits out of 16,384 (2% sparsity)    │
                         └────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────────────────────┐
                    │                INFERENCE PHASE                   │
                    └─────────────────────┬────────────────────────────┘
                                          │
     ┌────────────┐      ┌────────────┐   │   ┌────────────┐      ┌────────────┐
     │   Text     │ ───▶ │  Tokenize  │ ──┼─▶ │  Lookup    │ ───▶ │   Union    │
     │  "input"   │      │            │   │   │  Word FPs  │      │    SDRs    │
     └────────────┘      └────────────┘   │   └────────────┘      └────────────┘
                                          │                             │
                                          │                             ▼
                                          │                    ┌─────────────────┐
                                          │                    │ Text Fingerprint│
                                          │                    └─────────────────┘
                                          │                             │
                                          │                             ▼
                                          │                    ┌────────────────┐
                                          │                    │   Similarity   │
                                          │                    │   (AND count)  │
                                          ┴                    └────────────────┘
```

### Step 1: Text Processing

Text is tokenized using Unicode-aware word segmentation and normalized:

```
Input:  "The Quick Brown Fox!"
Output: ["the", "quick", "brown", "fox"]
```

Context windows capture local co-occurrence:

```
Text:   "the quick brown fox jumps"
Window: [("the", ["quick", "brown"]),
         ("quick", ["the", "brown", "fox"]),
         ("brown", ["the", "quick", "fox", "jumps"]),
         ...]
```

### Step 2: Self-Organizing Map Training

The SOM is a neural network that learns to map high-dimensional context vectors to a 2D grid:

1. **Initialize**: Create 128x128 grid of neurons with random weight vectors
2. **Present input**: For each context, create an embedding vector
3. **Find BMU**: Locate the neuron with the closest weights (Best Matching Unit)
4. **Update**: Move BMU and neighbors toward the input

```rust
// Training loop (simplified)
for (word, context) in training_contexts {
    let embedding = compute_embedding(&context);
    let bmu = som.find_bmu(&embedding);
    som.update(&embedding, bmu, learning_rate, neighborhood_radius);
}
```

The learning rate and neighborhood radius decay exponentially:
- Learning rate: 0.1 → 0.01
- Neighborhood radius: 64 → 1

### Step 3: Fingerprint Generation

After training, each word gets a fingerprint based on where its contexts map on the SOM:

```
Word: "king"
Contexts: ["the king said", "king of england", "crowned king", ...]
BMUs: [4523, 4524, 4589, 8901, ...]  # SOM positions for each context
Fingerprint: SDR with these positions set to 1
```

### Step 4: Similarity Computation

Text similarity is computed via fingerprint overlap:

```
Text A: "machine learning"  → SDR_A (328 active bits)
Text B: "deep learning"     → SDR_B (328 active bits)

Similarity = |SDR_A ∩ SDR_B| / √(|SDR_A| × |SDR_B|)  # Cosine
```

The overlap count is computed efficiently using RoaringBitmap's AND operation.

### Why This Works

1. **Distributional hypothesis**: Words appearing in similar contexts have similar meanings
2. **Topological organization**: SOM preserves neighborhood relationships
3. **Sparse coding**: High-dimensional semantic space compressed to interpretable binary patterns
4. **Efficiency**: Bitwise operations instead of floating-point math

## Configuration

### SOM Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dimension` | 128 | Grid size (dim × dim neurons) |
| `iterations` | 100,000 | Training iterations |
| `initial_learning_rate` | 0.1 | Starting learning rate |
| `final_learning_rate` | 0.01 | Ending learning rate |
| `initial_radius` | 64.0 | Starting neighborhood radius |
| `final_radius` | 1.0 | Ending neighborhood radius |
| `weight_dimension` | 300 | Embedding vector dimensionality |
| `context_window` | 5 | Context window size |
| `toroidal` | true | Wrap-around boundaries |

### Fingerprint Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sparsity` | 0.02 | Target fraction of active bits |
| `max_active_bits` | 328 | Maximum active bits (2% of 16,384) |
| `min_active_bits` | 10 | Minimum active bits |
| `weighted_aggregation` | true | Use TF-IDF weighting for text |

### Text Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lowercase` | true | Convert to lowercase |
| `min_token_length` | 2 | Minimum token length |
| `max_token_length` | 50 | Maximum token length |
| `remove_punctuation` | true | Strip punctuation |
| `remove_numbers` | false | Filter numeric tokens |
| `unicode_normalize` | true | Apply NFD normalization |

## Performance

### Benchmarks

On a corpus of 1 million sentences (modern x86-64, 8 cores):

| Operation | Time |
|-----------|------|
| Training (100K iterations) | ~5 minutes |
| Word fingerprint lookup | ~100 ns |
| Text fingerprinting (10 words) | ~2 μs |
| Similarity computation | ~50 ns |
| k-NN search (k=10, 100K vocab) | ~5 ms |

### Memory Usage

| Component | Size |
|-----------|------|
| SOM (128×128, 100-dim weights) | ~6.5 MB |
| Word fingerprint (328 bits) | ~50-100 bytes |
| Retina (100K words) | ~50 MB |
| Inverted index (100K words) | ~30 MB |

### Optimizations

- **SIMD f32**: Distance calculations use vectorized single-precision
- **Parallel BMU search**: Multi-threaded neuron scanning via Rayon
- **RoaringBitmap**: Compressed sparse sets with fast intersection
- **Memory mapping**: Large retinas loaded on-demand from disk
- **Batch training**: Accumulate updates before applying

## API Reference

### Core Types

```rust
// Main semantic model
pub struct Retina { ... }

impl Retina {
    pub fn load(path: &Path) -> Result<Self>;
    pub fn save(&self, path: &Path) -> Result<()>;
    pub fn fingerprint_text(&self, text: &str) -> Sdr;
    pub fn word_similarity(&self, w1: &str, w2: &str) -> Result<f64>;
    pub fn text_similarity(&self, t1: &str, t2: &str) -> f64;
    pub fn find_similar_words(&self, word: &str, k: usize) -> Result<Vec<(String, f64)>>;
    pub fn semantic_center(&self, words: &[&str]) -> Sdr;
    pub fn semantic_difference(&self, w1: &str, w2: &str) -> Option<Sdr>;
    pub fn analogy(&self, a: &str, b: &str, c: &str, k: usize) -> Result<Vec<(String, f64)>>;
}

// Sparse Distributed Representation
pub struct Sdr { ... }

impl Sdr {
    pub fn new(size: u32) -> Self;
    pub fn from_positions(positions: &[u32], size: u32) -> Self;
    pub fn cardinality(&self) -> u64;
    pub fn sparsity(&self) -> f64;
    pub fn overlap(&self, other: &Sdr) -> Sdr;  // AND
    pub fn union(&self, other: &Sdr) -> Sdr;    // OR
    pub fn xor(&self, other: &Sdr) -> Sdr;      // XOR
    pub fn jaccard_similarity(&self, other: &Sdr) -> f64;
    pub fn cosine_similarity(&self, other: &Sdr) -> f64;
    pub fn overlap_similarity(&self, other: &Sdr) -> f64;
}

// Self-Organizing Map
pub struct Som { ... }

impl Som {
    pub fn new(config: &SomConfig) -> Self;
    pub fn find_bmu(&self, input: &[f64]) -> Result<usize>;
    pub fn find_bmu_parallel(&self, input: &[f64]) -> Result<usize>;
    pub fn update(&mut self, input: &[f64], bmu: usize, lr: f64, radius: f64);
}

// Sentence segmentation
pub struct SentenceSegmenter { ... }

impl SentenceSegmenter {
    pub fn new(model: &str) -> Result<Self>;  // "sat-3l-sm", "sat-3l", "sat-6l", "sat-12l"
    pub fn segment(&mut self, text: &str) -> Result<Vec<String>>;
}

// Semantic segmentation (topic detection)
pub struct SemanticSegmenter { ... }

impl SemanticSegmenter {
    pub fn new(retina: Retina) -> Result<Self>;
    pub fn with_config(retina: Retina, config: SegmentationConfig) -> Result<Self>;
    pub fn segment(&mut self, text: &str) -> Result<SegmentationResult>;
    pub fn segment_text(&mut self, text: &str) -> Result<Vec<SemanticSegment>>;
}

pub struct SegmentationResult {
    pub segments: Vec<SemanticSegment>,
    pub boundary_indices: Vec<usize>,
    pub similarity_scores: Vec<f32>,
    pub depth_scores: Vec<f32>,
}
```

### Advanced Index Types

```rust
// HNSW Graph Index
pub struct HnswIndex { ... }

impl HnswIndex {
    pub fn build(config: HnswConfig, data: Vec<Sdr>) -> Self;
    pub fn search(&self, query: &Sdr, k: usize) -> Vec<(u32, f32)>;
    pub fn search_ef(&self, query: &Sdr, k: usize, ef: usize) -> Vec<(u32, f32)>;
    pub fn stats(&self) -> HnswStats;
}

pub struct HnswConfig {
    pub m: usize,              // Max connections at layer 0
    pub m_max: usize,          // Max connections at higher layers
    pub ef_construction: usize, // Build-time candidate list
    pub ef_search: usize,       // Query-time candidate list
    pub ml: f64,               // Level multiplier
    pub seed: Option<u64>,
}

// Product Quantization
pub struct ProductQuantizer { ... }

impl ProductQuantizer {
    pub fn train(vectors: &[Vec<f32>], config: PqConfig) -> Self;
    pub fn encode(&self, vector: &[f32]) -> Vec<u8>;
    pub fn decode(&self, codes: &[u8]) -> Vec<f32>;
    pub fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32;
}

// Locality-Sensitive Hashing
pub trait LshHasher: Clone {
    fn hash(&self, data: &Sdr) -> u64;
    fn hash_band(&self, data: &Sdr, band: usize) -> u64;
}

pub struct SimHash { ... }      // Cosine similarity
pub struct MinHash { ... }      // Jaccard similarity
pub struct BitSamplingLsh { ... } // Binary vector hashing

pub struct LshIndex<H: LshHasher> { ... }

impl<H: LshHasher> LshIndex<H> {
    pub fn new(hasher: H, num_tables: usize, bands_per_table: usize) -> Self;
    pub fn build(&mut self, data: &[Sdr]);
    pub fn query(&self, query: &Sdr) -> Vec<u32>;
}

// Hyperdimensional Computing
pub struct Hypervector { ... }

impl Hypervector {
    pub fn random(dimension: u32, sparsity: f64, seed: Option<u64>) -> Self;
    pub fn bundle(vectors: &[&Hypervector], threshold: f64) -> Self;
    pub fn bind(&self, other: &Hypervector) -> Self;
    pub fn unbind(&self, other: &Hypervector) -> Self;
    pub fn permute(&self, amount: i32) -> Self;
    pub fn similarity(&self, other: &Hypervector) -> f64;
}

pub struct ItemMemory { ... }
pub struct SequenceEncoder { ... }
pub struct NgramEncoder { ... }
pub struct AnalogySolver { ... }

// Mini-batch SOM Training
pub struct BatchSomTrainer { ... }

impl BatchSomTrainer {
    pub fn new(config: BatchSomConfig) -> Self;
    pub fn train(&mut self, som: &mut Som, data: &[Vec<f32>]) -> TrainingMetrics;
}

pub struct TrainingMetrics {
    pub quantization_error: f32,
    pub topographic_error: f32,
    pub learning_rate_history: Vec<f32>,
}
```

### Similarity Measures

```rust
pub trait SimilarityMeasure {
    fn similarity(&self, a: &Sdr, b: &Sdr) -> f64;
    fn distance(&self, a: &Sdr, b: &Sdr) -> f64 { 1.0 - self.similarity(a, b) }
}

pub struct CosineSimilarity;    // |A ∩ B| / √(|A| × |B|)
pub struct JaccardSimilarity;   // |A ∩ B| / |A ∪ B|
pub struct OverlapSimilarity;   // |A ∩ B| / min(|A|, |B|)
```

## Testing

Run the complete test suite:

```bash
# All tests (109 total: 96 unit + 13 integration)
cargo test

# With output
cargo test -- --nocapture

# Specific module
cargo test som::
cargo test fingerprint::

# Integration tests only
cargo test --test integration_tests
```

Test coverage includes:
- Unit tests for all modules
- Integration tests for end-to-end workflows
- Property-based testing with proptest
- Snapshot testing with insta

## Scaling to Terabytes & Cross-Lingual Alignment

This section documents the architectural considerations and future roadmap for scaling Proteus to terabyte-scale corpora and enabling cross-lingual semantic fingerprinting.

### Current Limitations

The current implementation loads all training data into memory:

```rust
// Current approach - O(corpus_size) memory
let documents: Vec<String> = reader.lines().collect();
for doc in &documents {
    for (center, context) in tokenizer.context_windows(doc, 2) {
        contexts.push((center, context));  // All contexts in RAM
    }
}
```

This works well for corpora up to ~10GB but becomes impractical for terabyte-scale training.

### Terabyte-Scale Training Architecture

#### 1. Streaming Context Extraction

Replace in-memory context storage with a streaming pipeline:

```rust
/// Streaming iterator over sharded context files
trait ContextStream: Iterator<Item = (String, Vec<String>)> {
    fn shuffle_window(&mut self, buffer_size: usize);
    fn sample(&mut self, rate: f64) -> impl ContextStream;
}

/// Phase 1: Extract contexts to sharded files on disk
struct ContextExtractor {
    output_dir: PathBuf,
    shard_size: usize,  // e.g., 10M contexts per shard
    current_shard: BufWriter<File>,
}

/// Phase 2: Memory-mapped streaming with shuffle buffer
struct ShardedContextReader {
    shards: Vec<PathBuf>,
    shuffle_buffer: Vec<(String, Vec<String>)>,
    buffer_size: usize,  // e.g., 1M contexts for local shuffling
}
```

#### 2. Distributed Word Embeddings

Use deterministic hashing for consistent random vectors across workers:

```rust
/// Same word always gets same vector - no coordination needed
fn word_to_random_vector(word: &str, dim: usize, global_seed: u64) -> Vec<f32> {
    let seed = hash(global_seed, word);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}
```

This enables distributed workers to compute identical embeddings independently.

#### 3. Hierarchical SOM Training

For massive vocabularies, use a multi-resolution SOM hierarchy:

```rust
struct HierarchicalSom {
    /// Level 0: Coarse clusters (32×32 = 1K neurons)
    level0: Som,

    /// Level 1: Per-cluster refinement (each has 64×64 local SOM)
    level1: HashMap<usize, Som>,

    // Final fingerprint = [level0_pos * 4096 + level1_pos]
    // Provides 1K × 4K = 4M effective positions
}
```

#### 4. Distributed Training Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                     Coordinator Node                             │
│  - Tracks global vocabulary statistics                          │
│  - Aggregates SOM weight updates (parameter server)             │
│  - Schedules training batches across workers                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Worker 1    │   │   Worker 2    │   │   Worker N    │
│  Shards 1-100 │   │ Shards 101-200│   │    ...        │
│               │   │               │   │               │
│ 1. Stream ctx │   │ 1. Stream ctx │   │               │
│ 2. Local emb  │   │ 2. Local emb  │   │               │
│ 3. Find BMUs  │   │ 3. Find BMUs  │   │               │
│ 4. Send deltas│   │ 4. Send deltas│   │               │
└───────────────┘   └───────────────┘   └───────────────┘
```

Key techniques:
- **Hogwild SGD**: Approximate asynchronous SOM updates
- **Minibatch accumulation**: Collect BMU statistics before weight sync
- **Vocabulary filtering**: Discard words with frequency < threshold

Target: 1TB corpus processed in ~24 hours on a 10-node cluster.

#### 5. Incremental Learning

Add new vocabulary without full retraining:

```rust
impl Retina {
    /// Expand vocabulary from additional corpus
    fn incremental_train(
        &mut self,
        new_contexts: impl ContextStream,
        frozen_som: bool,  // If true, only assign BMUs (no weight updates)
    ) -> Result<()> {
        // 1. For new words: assign BMUs based on context similarity
        // 2. For existing words: update BMU statistics
        // 3. Optionally fine-tune SOM with small learning rate
        // 4. Recompute fingerprints for affected words
    }
}
```

### Cross-Lingual Fingerprint Alignment

The goal is semantic equivalence across languages:
```
fingerprint("king", english_retina) ≈ fingerprint("roi", french_retina)
```

#### The Alignment Problem

Independently trained language models have:
- **Isomorphic structure**: Same underlying concepts exist
- **Arbitrary orientation**: Different coordinate systems
- **Scale differences**: Varying corpus statistics

#### Approach 1: Post-hoc Alignment (Procrustes)

Train separately, then align using anchor pairs:

```rust
/// Orthogonal Procrustes alignment between SOM spaces
struct SomAlignment {
    rotation: Matrix,      // Rotation/reflection (dim × dim)
    translation: Vec<f64>, // Translation vector
    scale: f64,            // Scaling factor
}

impl SomAlignment {
    /// Learn alignment from bilingual dictionary
    fn from_anchors(
        source_retina: &Retina,
        target_retina: &Retina,
        anchors: &[(String, String)],  // e.g., [("king", "roi"), ("water", "eau"), ...]
    ) -> Result<Self> {
        // 1. Extract centroid positions for anchor words
        let source_positions = anchors.iter()
            .filter_map(|(src, _)| source_retina.get_centroid(src))
            .collect();
        let target_positions = anchors.iter()
            .filter_map(|(_, tgt)| target_retina.get_centroid(tgt))
            .collect();

        // 2. Solve: min ||s * R * source + t - target||²
        //    Subject to: R is orthogonal
        procrustes_align(&source_positions, &target_positions)
    }

    /// Transform fingerprint from source to aligned target space
    fn transform(&self, fp: &Sdr, grid_dim: u32) -> Sdr {
        let transformed: Vec<u32> = fp.iter()
            .map(|pos| {
                let (row, col) = (pos / grid_dim, pos % grid_dim);
                let (new_row, new_col) = self.transform_point(row, col);
                (new_row * grid_dim + new_col).clamp(0, grid_dim * grid_dim - 1)
            })
            .collect();
        Sdr::from_positions(&transformed, grid_dim * grid_dim)
    }
}
```

Requires ~5,000-10,000 anchor pairs per language pair.

#### Approach 2: Joint Multilingual Training (Recommended)

Train a single shared semantic space across all languages:

```rust
struct MultilingualTrainer {
    /// Shared SOM - all languages map to same grid
    shared_som: Som,

    /// Language-specific tokenizers
    tokenizers: HashMap<LangCode, Tokenizer>,

    /// Parallel corpus provides alignment signal
    parallel_corpus: Vec<ParallelSentence>,
}

struct ParallelSentence {
    // Same meaning expressed in different languages
    sentences: HashMap<LangCode, String>,
}

impl MultilingualTrainer {
    fn train(&mut self, corpora: HashMap<LangCode, impl ContextStream>) {
        // Phase 1: Interleaved monolingual training
        // - Sample contexts from all languages uniformly
        // - Train shared SOM with mixed input
        // - Words cluster by meaning, not by language

        // Phase 2: Parallel corpus alignment
        for parallel in &self.parallel_corpus {
            // Extract contexts from translation pairs
            // Apply soft constraint: translations → similar BMUs
            self.align_translations(parallel);
        }

        // Phase 3: Cross-lingual analogy preservation
        // Verify: king:queen :: roi:reine :: könig:königin
    }

    fn align_translations(&mut self, parallel: &ParallelSentence) {
        // Cross-lingual loss: penalize BMU distance between translations
        // Modified SOM update pulls translation pairs together
    }
}
```

#### Cross-Lingual Training Signal

The key insight: **parallel corpora provide natural alignment**

```rust
/// Loss function during training
fn cross_lingual_loss(
    word_en: &str,
    word_fr: &str,  // Translation of word_en
    bmu_en: usize,
    bmu_fr: usize,
    som_dimension: usize,
) -> f64 {
    let (row_en, col_en) = (bmu_en / som_dimension, bmu_en % som_dimension);
    let (row_fr, col_fr) = (bmu_fr / som_dimension, bmu_fr % som_dimension);

    // Squared distance on SOM grid
    (row_en as f64 - row_fr as f64).powi(2) +
    (col_en as f64 - col_fr as f64).powi(2)
}
```

#### Multilingual Retina Format

```rust
struct MultilingualRetina {
    /// Shared semantic grid (all languages)
    grid_dimension: u32,

    /// Per-language vocabularies mapped to shared grid
    languages: HashMap<LangCode, LanguageData>,

    /// Unified inverted index (positions → words from all languages)
    shared_index: InvertedIndex,
}

struct LanguageData {
    fingerprints: HashMap<String, WordFingerprint>,
    tokenizer_config: TokenizerConfig,
    stopwords: HashSet<String>,
}

impl MultilingualRetina {
    /// Fingerprint text in any supported language
    fn fingerprint(&self, text: &str, lang: LangCode) -> Sdr {
        let lang_data = &self.languages[&lang];
        // Language-specific tokenization, shared fingerprint space
        self.fingerprint_with_tokenizer(text, &lang_data.tokenizer_config)
    }

    /// Cross-lingual similarity (e.g., English query vs French document)
    fn cross_lingual_similarity(
        &self,
        text1: &str, lang1: LangCode,
        text2: &str, lang2: LangCode,
    ) -> f64 {
        let fp1 = self.fingerprint(text1, lang1);
        let fp2 = self.fingerprint(text2, lang2);
        fp1.cosine_similarity(&fp2)  // Works because shared space!
    }

    /// Find translations (words in target language with similar fingerprints)
    fn find_translations(
        &self,
        word: &str,
        source_lang: LangCode,
        target_lang: LangCode,
        k: usize
    ) -> Vec<(String, f64)> {
        let source_fp = self.languages[&source_lang].fingerprints.get(word)?;
        self.search_in_language(&source_fp.fingerprint, target_lang, k)
    }
}
```

#### Alignment Data Sources

| Source | Pairs | Languages | Quality |
|--------|-------|-----------|---------|
| Wikipedia parallel titles | 50M+ | 300+ | Medium |
| Wiktionary translations | 10M+ | 100+ | High |
| OPUS parallel corpora | 100B+ tokens | 100+ | Varies |
| Panlex | 25M+ | 5,700+ | Medium |
| MUSE bilingual dictionaries | 100K+ per pair | 100+ | Curated |

#### Why Semantic Folding Enables Cross-Lingual Alignment

The distributional hypothesis transcends language boundaries:
- **"roi"** in French appears in contexts similar to **"king"** in English
- Both words co-occur with: governance, crown, throne, palace, reign...
- SOM training naturally clusters words by contextual usage
- Parallel corpora provide the bridge to align coordinate systems

This is fundamentally different from (and complementary to) approaches like:
- **Word2Vec/FastText alignment**: Requires large parallel corpora
- **Multilingual BERT**: Requires massive compute for pretraining
- **Dictionary-based MT**: Limited by dictionary coverage

SDR alignment preserves the interpretability and efficiency of semantic folding while enabling cross-lingual applications.

### Implementation Roadmap

#### Phase 1: Streaming Infrastructure
- [ ] Implement `ContextExtractor` for sharded disk output
- [ ] Implement `ShardedContextReader` with shuffle buffer
- [ ] Add memory-mapped context file format
- [ ] Target: 100M contexts/hour on single node

#### Phase 2: Distributed Training
- [ ] Implement deterministic word→vector hashing
- [ ] Add vocabulary filtering (min frequency threshold)
- [ ] Implement distributed BMU computation
- [ ] Add parameter server for SOM weight aggregation
- [ ] Target: 1TB corpus in 24 hours on 10 nodes

#### Phase 3: Cross-Lingual Foundation
- [ ] Define `MultilingualRetina` format and serialization
- [ ] Implement shared SOM training with language interleaving
- [ ] Add parallel corpus loading (OPUS format)
- [ ] Implement cross-lingual loss function

#### Phase 4: Alignment Refinement
- [ ] Implement Procrustes post-hoc alignment (fallback method)
- [ ] Add bilingual dictionary support (MUSE format)
- [ ] Implement cross-lingual analogy evaluation
- [ ] Target: >70% precision@1 on MUSE benchmark

#### Phase 5: Production Features
- [ ] Incremental vocabulary updates
- [ ] Language detection + automatic routing
- [ ] Quantized fingerprints for mobile/edge deployment
- [ ] REST API for cross-lingual semantic search

### Research Challenges

Several open questions remain for large-scale cross-lingual deployment:

1. **Topology Preservation**: Does the SOM's 2D topology transfer meaningfully across languages with different morphological structures?

2. **Rare Word Handling**: Low-frequency words have noisy fingerprints; this is exacerbated in multilingual settings where corpus sizes vary by language.

3. **Polysemy**: "bank" (river) vs "bank" (financial) - different meanings need different fingerprints, but translations may disambiguate differently.

4. **Script Differences**: Aligning Latin, Cyrillic, Arabic, and CJK character systems requires careful tokenization strategies.

5. **Morphological Complexity**: Agglutinative languages (Turkish, Finnish) vs isolating languages (English, Chinese) have vastly different word/morpheme ratios.

## Contributing

Contributions are welcome! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone and build
git clone https://github.com/proteus/proteus.git
cd proteus
cargo build

# Run tests
cargo test

# Run benchmarks
cargo bench

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy -- -W clippy::all
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Cortical.io** - For pioneering Semantic Folding technology and publishing research that made this implementation possible
- **wtpsplit** - For the excellent sentence segmentation models
- **RoaringBitmap** - For efficient compressed bitmap implementation
- **Rayon** - For effortless parallelism in Rust

## References

### Semantic Folding & SDRs
1. Webber, F. (2015). "Semantic Folding Theory And its Application in Semantic Fingerprinting." arXiv:1511.08855
2. De Sousa Webber, F. (2015). "The Semantic Folding Theory And its Application in Semantic Fingerprinting." Cortical.io
3. Kohonen, T. (2001). "Self-Organizing Maps." Springer Series in Information Sciences.
4. Ahmad, S., & Hawkins, J. (2016). "How do neurons operate on sparse distributed representations? A mathematical theory of sparsity, neurons and active dendrites." arXiv:1601.00720
5. Hearst, M. A. (1997). "TextTiling: Segmenting Text into Multi-paragraph Subtopic Passages." Computational Linguistics, 23(1), 33-64.

### Advanced Indexing Algorithms
6. Malkov, Y. A., & Yashunin, D. A. (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." IEEE TPAMI. arXiv:1603.09320
7. Jégou, H., Douze, M., & Schmid, C. (2011). "Product Quantization for Nearest Neighbor Search." IEEE TPAMI.
8. Charikar, M. S. (2002). "Similarity Estimation Techniques from Rounding Algorithms." STOC.
9. Broder, A. Z. (1997). "On the Resemblance and Containment of Documents." SEQUENCES.

### Hyperdimensional Computing
10. Kanerva, P. (2009). "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors." Cognitive Computation, 1(2), 139-159.
11. Plate, T. A. (2003). "Holographic Reduced Representation: Distributed Representation for Cognitive Structures." CSLI Publications.
12. Gayler, R. W. (2003). "Vector Symbolic Architectures Answer Jackendoff's Challenges for Cognitive Neuroscience." ICCS/ASCS Joint International Conference.

---

*Proteus: Named after the Greek god who could assume any form - just as SDRs can represent any semantic concept.*
