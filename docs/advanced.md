# Advanced search and vector APIs

[Documentation index](../README.md#documentation) · [Library guide](library.md)

These modules are public library interfaces. They are not selected automatically
by `proteus train` or `proteus similar`. The latter uses the retina's cosine
search path. Source links below provide the complete signatures and tests.

## Index selection

| Interface | Representation / result | Main controls |
| --- | --- | --- |
| [`InvertedIndex`](../src/index/inverted.rs) | Bitmap postings; raw overlap counts | Grid size and candidate count; `Retina` adds exact cosine ranking |
| [`HnswIndex`](../src/index/hnsw.rs) | Binary SDRs; `(input_id, Jaccard distance)` | Graph degree, construction/search breadth and seed |
| [`LshIndex`](../src/index/lsh.rs) | Hash buckets; candidate ID set | Hasher, table count and band count; callers rerank candidates |
| [`ProductQuantizer`](../src/index/pq.rs) | Dense `f32` vectors; byte codes and approximate squared distances | Subquantizers, centroids and k-means iterations |
| [`OptimizedProductQuantizer`](../src/index/pq.rs) | Dense-vector preprocessing plus PQ | OPQ iteration count and PQ configuration |
| [`SemanticLookupEngine`](../src/index/semantic_lookup.rs) | Grid positions; scored words and factor breakdowns | Gaussian width, radius, adaptive expansion and clusters |

An approximate index can miss a true neighbour. Compare recall against exhaustive
search using the same metric; HNSW Jaccard distances cannot be substituted for
cosine scores without changing the objective. No fixed speedup or compression
ratio is guaranteed across corpora.

## HNSW and LSH

```rust
use proteus::{HnswConfig, HnswIndex, Sdr};
use proteus::index::create_minhash_index;

let fingerprints = vec![
    Sdr::from_positions(&[1, 2, 3], 256),
    Sdr::from_positions(&[2, 3, 4], 256),
];
let query = fingerprints[0].clone();
let hnsw = HnswIndex::build(
    HnswConfig { seed: Some(42), ..Default::default() }, fingerprints.clone()
);
let neighbours = hnsw.search(&query, 2);

let mut lsh = create_minhash_index(64, 4, Some(42));
for (id, fp) in fingerprints.iter().enumerate() {
    lsh.insert(id as u32, fp);
}
let candidates = lsh.query(&query);
```

Keep the input-ID-to-word mapping outside these indexes. `SimHash` supports
dense-vector hashing; `MinHash::hash_sdr` and `BitSamplingLsh::hash` support SDRs.
Direct `LshIndex::new` takes a vector of hashers, a table count and a band count.

## Product quantization

```rust
use proteus::{PqConfig, ProductQuantizer};

let vectors = vec![vec![0.0, 1.0, 0.0, 1.0], vec![1.0, 0.0, 1.0, 0.0]];
let pq = ProductQuantizer::train(PqConfig {
    num_subquantizers: 2, num_centroids: 2, kmeans_iterations: 5,
    ..Default::default()
}, &vectors);
let codes = pq.encode_batch(&vectors);
let table = pq.compute_distance_table(&vectors[0]);
let distance = pq.asymmetric_distance(&table, &codes[1]);
let reconstructed = pq.decode(&codes[0]);
```

Inputs must be nonempty, finite, and have equal dimensions divisible by the
positive subquantizer count. Codes store centroid IDs as `u8`; use at most 256
centroids. This is dense-vector quantization, not compression of `.retina` files.

## Batch SOM and growing maps

[`BatchSomTrainer`](../src/som/batch.rs) is separate from the reference online
trainer. Its constructor is `new(config, grid_dim, weight_dim)` and `train` takes
a mutable flat `f32` weight array plus labelled vectors:

```rust
use proteus::{BatchSomConfig, BatchSomTrainer};

let mut trainer = BatchSomTrainer::new(BatchSomConfig {
    epochs: 1, warm_restarts: 0, seed: Some(42), ..Default::default()
}, 4, 2);
let mut weights = vec![0.0_f32; 4 * 4 * 2];
let samples = vec![("sample".to_owned(), vec![0.5_f32, 1.0])];
let assignments = trainer.train(&mut weights, &samples);
let metrics = trainer.metrics();
```

`BatchSomConfig` exposes batch size, Adam parameters, learning/radius schedules
and optional cosine annealing with restarts. `GrowingSom` exposes a separate map
interface. Neither carries the online trainer's determinism or final-assignment
contract by implication. Supply valid nonempty inputs and schedule parameters.

## Hyperdimensional computing

[`Hypervector`](../src/fingerprint/hdc.rs) provides thresholded bundling, XOR
binding/unbinding and cyclic permutation. `ItemMemory`, `SequenceEncoder`,
`NgramEncoder` and `AnalogySolver` build symbolic operations on these vectors.

```rust
use proteus::Hypervector;

let item = Hypervector::random(1024, 0.5, Some(1));
let role = Hypervector::random(1024, 0.5, Some(2));
let bound = item.bind(&role);
let recovered = bound.unbind(&role);
assert_eq!(recovered.sdr, item.sdr);
```

Random item vectors do not acquire corpus semantics merely by binding them.
`AnalogySolver::new` takes an `ItemMemory`; the corpus-backed signed analogy
method is `Retina::analogy`. These are distinct APIs.

Algorithm references: [HNSW](https://arxiv.org/abs/1603.09320),
[product quantization](https://doi.org/10.1109/TPAMI.2010.57), and
[Kanerva's HDC overview](https://doi.org/10.1007/s12559-009-9009-8).
