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
│   │   └── training.rs     # Training algorithms
│   │
│   ├── fingerprint/        # Fingerprint Generation
│   │   ├── mod.rs
│   │   ├── sdr.rs          # SDR with RoaringBitmap
│   │   ├── word.rs         # Word fingerprinting
│   │   └── text.rs         # Text fingerprinting
│   │
│   ├── roaring/            # Vendored RoaringBitmap
│   │
│   ├── storage/            # Persistence
│   │   ├── mod.rs
│   │   ├── format.rs       # Binary format (header + index + data)
│   │   └── retina.rs       # Retina management
│   │
│   ├── index/              # Search Index
│   │   ├── mod.rs
│   │   └── inverted.rs     # Inverted index for similarity search
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

1. Webber, F. (2015). "Semantic Folding Theory And its Application in Semantic Fingerprinting." arXiv:1511.08855
2. De Sousa Webber, F. (2015). "The Semantic Folding Theory And its Application in Semantic Fingerprinting." Cortical.io
3. Kohonen, T. (2001). "Self-Organizing Maps." Springer Series in Information Sciences.
4. Ahmad, S., & Hawkins, J. (2016). "How do neurons operate on sparse distributed representations? A mathematical theory of sparsity, neurons and active dendrites." arXiv:1601.00720

---

*Proteus: Named after the Greek god who could assume any form - just as SDRs can represent any semantic concept.*
