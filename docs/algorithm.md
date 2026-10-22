# Algorithm and validation contract

Proteus implements the context-map architecture described in Webber's Semantic
Folding paper, pp. 31-32 (PDF pages 30-31). It is an independent implementation,
not a reproduction of the proprietary Cortical.io training algorithm or retina.
The paper specifies mapping snippets and folding their word incidence into bits;
it does not prescribe the numerical feature representation, map optimizer,
initialization, or every sparsification choice used here.

## Assumption register

| ID | Assumption / explicit implementation choice | Falsification probe |
| --- | --- | --- |
| A1 | A bounded paragraph fragment represents one useful context. | Compare paragraph/line modes and snippet lengths on a domain-labelled validation set. Hard-wrapped lines must remain within paragraphs. |
| A2 | TF-IDF Euclidean geometry captures enough topical similarity for folding. | Measure held-out neighbour quality against a TF-IDF baseline; the controlled topic test is only a mechanics check. |
| A3 | The reference corpus contains the intended senses. | Inspect occurrences of the query and evaluation coverage. More updates cannot supply absent senses. |
| A4 | Optional random projection retains adequate geometry. | Run identical corpus/seed with `projection_dim=0` and measure distance/ranking changes. No fixed distortion guarantee is claimed for an arbitrary dimension. |
| A5 | Low-cardinality fingerprints contain sufficient evidence. | Stratify evaluation by token count and number of distinct occupied cells. Never pad sparse words with invented cells. |
| A6 | Reproducibility means identical inputs/configuration, binary, platform and arithmetic backend. | Compare complete saved bytes and query results across fresh runs and Rayon thread counts. Cross-architecture floating-point identity is not claimed. |

Semantic-quality conclusions depend on A1-A5. Determinism depends on A6.
The quoted user's actual corpus and retina were not present in the workspace;
their content, version and empirical quality are unknown.

## Contexts and features

The UTF-8 reader propagates all errors. Paragraph mode joins physical lines until
a blank line. Line mode treats each nonempty physical line as one document.
Within either document, consecutive normalized tokens are partitioned into
nonoverlapping snippets of at most `snippet_tokens` tokens (default 64). These
are bounded token windows, not neural sentence boundaries. No context crosses a
document boundary. Unicode normalization, case handling and token filters are
shared by training and inference and saved in the model.
The retained normalization policy uses NFD, UTF-8 byte-length limits and ASCII
punctuation removal. Non-ASCII apostrophes can remain within word tokens;
there is no stemming, lemmatization or automatic singular/plural merging.

Let `M` be the number of nonempty snippets, `V` the vocabulary meeting
`min_count`, and `df(j)` the number of snippets containing word `j`. A word is a
feature only if `df(j)/M <= max_document_frequency`. This feature filter does not
remove that word from the output vocabulary. Feature IDs follow lexical order.

For snippet `s` and retained feature `j` with token count `c(s,j) > 0`:

```text
idf(j) = 1 + ln((1 + M)/(1 + df(j)))
v(s,j) = (1 + ln(c(s,j))) * idf(j)
x(s)   = v(s) / ||v(s)||_2
```

All coordinates are dimensionless. Empty feature vectors are omitted and counted
in diagnostics; words seen only in omitted snippets have no fingerprint.
Word frequencies count tokens, not training draws, and are normalized over
the vocabulary meeting `min_count` before the zero-feature snippet filtering.

The exact default uses `D = number of retained features`. Optional projection
constructs `R(j,k)` independently as `+1/sqrt(D)` or `-1/sqrt(D)` with equal
probability from a seeded ChaCha8 generator, then normalizes `v R`. For a fixed
vector `v`, independent signs give `E[||v R||^2] = ||v||^2`: cross terms have
expectation zero. This expectation is not a per-corpus error bound. Zero projected
vectors are omitted. Projection and map initialization use distinct seed streams.

## Online SOM

There are `G = dimension^2` cells. Prototypes initialize from seeded draws of
actual snippet vectors. The trainer makes exactly `T = iterations` online
updates. It shuffles input indices once per epoch, without replacement within
each epoch; the final epoch may be incomplete.

```text
for t in 0 .. T-1:
    x = next shuffled snippet vector
    b = argmin_i sum_j (x[j] - w[i,j])^2   # ties: smallest cell index
    q = t/(T-1) if T > 1 else 0
    alpha = exp((1-q)*ln(alpha_initial) + q*ln(alpha_final))
    sigma = exp((1-q)*ln(sigma_initial) + q*ln(sigma_final))
    for every cell i, exactly once:
        h = exp(-distance_grid(i,b)^2/(2*sigma^2))
        w[i] += alpha*h*(x - w[i])
freeze all prototypes
assign every usable snippet to its final-map BMU
```

Grid distances use cell units. On a torus, each axis distance is
`min(abs(a-b), dimension-abs(a-b))`. Planar mode uses ordinary axis distance.
`sigma` is in cells, `alpha` and `h` are dimensionless. Every prototype coordinate
remains dimensionless. No neighborhood truncation, repeated toroidal update,
stale BMU batch, approximate hierarchy, GPU reduction, or hidden iteration cap
is used. Progress callbacks report online updates; final assignment is an
additional pass. The legacy `train`, `train_fast` and progress adapters use this
same optimizer. The legacy `WordEmbeddings` adapter is random indexing, not
skip-gram; new corpus training does not use it.

Weights, inputs and distances use IEEE 754 binary32 arithmetic; schedules and
reported quantization error use binary64. The Gaussian is evaluated as the
product of its two one-dimensional axis factors, then rounded to binary32.
Exact BMU means exhaustive minimization
of computed distances, not exact real arithmetic. The scalar-oracle test checks
absolute weight error below `3e-7` on its finite fixture, including radii larger
than the torus. This is a test tolerance, not a universal training error bound.

## Folding, composition and search

For each final assigned snippet, increment each **distinct** word's count at its
cell once. A word's candidate bits are exactly its observed cells. If necessary,
retain the highest-count cells with cell-index tie-breaking, up to:

```text
K = min(ceil(sparsity * G), max_active_bits, G)
```

For example, `ceil(0.02 * 128^2) = ceil(327.68) = 328` cells. Sparse evidence can
produce fewer than `K` bits. `min_active_bits` is retained for API compatibility;
it never fabricates evidence. A density cap is an explicit engineering addition
to the paper's word-incidence description and can lose rare senses (A5).

Text aggregation counts word-bit occurrences and retains the highest counts;
ties use cell index. The optional union mode caps its union deterministically.
Cosine similarity is `|A intersect B| / sqrt(|A| * |B|)`, with zero returned if
either set is empty. Results sort by descending similarity, then lexical word.
There is no query-time stopword filter. An unknown word returns an error;
unknown text tokens supply no bits. All-unknown text produces an empty SDR.

The inverted index enumerates **all** positive-overlap candidates before cosine
ranking. Cutting candidates by raw overlap changes the objective when bit counts
vary. Zero-overlap entries are included when required to match exhaustive top-k.
Analogy computes cosine against the signed vector `B + C - A`; this is a defined
arithmetic operation, not a validated claim about linguistic analogy quality.

## Persistence

Version 4 retains the 64-byte header and adds JSON metadata before the word index,
with an optional memory-mappable posting index after the string pool.
Metadata includes training parameters, resolved seed, diagnostics, the SHA-256
of normalized length-prefixed snippet/token sequences, inference configuration,
word frequencies and POS tags. Bitmaps and string/index records retain their
previous layout. Canonical ordering makes same-environment seeded models byte
reproducible. Header version, section bounds, UTF-8, duplicate keys, bitmap
structure, bit universes and persisted-index agreement with fingerprints are validated by ordinary and mapped readers.
Writes use an atomic replacement from a temporary file in the destination
directory. A mapped backing file must not be modified externally in place.

Upstream versions 1, 2 (bincode index), 3 (mmap index), and the earlier local
metadata-only version 2 remain readable. Legacy layouts are distinguished by
version, flags and the index offset. Version 1 remains readable. Its missing metadata cannot be recovered: frequencies
default to zero, inference uses legacy defaults, and training provenance is
reported as unknown. An old model's learned geometry is unchanged by loading it
with the new code; it needs retraining to obtain the new algorithm.

## Cost and bounded extensions

Let `N` be input tokens, `M` usable snippets, `F` retained features, `D` final
vector dimension, `G` cells, and `T` updates. Ignoring tokenization and ordered-map
logarithmic factors, dense training/assignment costs `O((T+M)*G*D)` and holds
`O(N + (M+G)*D + F*D)` memory with projection, or `O(N + (M+G)*F)` without it.
The projection matrix is absent in exact mode. Dense f32 inputs need `4*M*D`
bytes. The current f64 map plus f32 working copy need `12*G*D` bytes, excluding
allocation overhead. Exact search scans vocabulary fingerprints and sorts results;
index search costs posting-list traversal plus candidate sorting. No terabyte
streaming, measured throughput, or distributed training claim is made.

| Impact | Pattern / opportunity | Bound |
| --- | --- | --- |
| High | A 128x128 grid with few contexts has poorly supported cells; more updates do not increase evidence. | Report occupied cells, quantify evaluation by support, or use a smaller grid. |
| High | A domain-specific book supplies domain-specific senses. | General-English neighbour quality requires an independently evaluated broader corpus (A3). |
| High | Exact vocabulary geometry can be expensive. | Opt-in projection reduces `D`; validate the quality/cost trade-off (A4). |
| Medium | Hard-wrapped text and unrelated line documents need different boundaries. | Use the explicit input mode and verify snippets (A1). |
| Medium | Capping fingerprints can discard rare senses. | Evaluate uncapped or larger-density models rather than interpreting a missing bit as an absent sense (A5). |
| Low | Binary analogy arithmetic is not a linguistic guarantee. | Keep arithmetic tests separate from a domain analogy benchmark. |

## Self-review and quality gates

- QG1: This contract describes implementation and evidence; no normative decision is required.
- QG2: A1-A6 identify assumptions, dependent conclusions and falsification probes.
- QG3: Tests cover online updates, final assignment, deterministic training/querying,
  input boundaries, index equivalence, malformed files and legacy compatibility.
- QG4: Equations, units, complexity and fixture tolerances are specified above.
- QG5: Zero vectors, unknown words, small corpora, ties, zero/large k, invalid
  configuration, corrupt offsets and missing legacy metadata have explicit behavior.
- QG6: References below establish the published architecture and algorithm family;
  implementation choices and unmeasured semantic quality are identified separately.
- QG7: Bounded opportunities/risks are listed above. Controlled tests do not establish
  general-English performance or equivalence to a proprietary model.

## Primary references

1. Webber, F. *Semantic Folding Theory and its Application in Semantic Fingerprinting*,
   arXiv:1511.08855v2 (2016), particularly pp. 31-32 and 36-37.
   <https://arxiv.org/abs/1511.08855>
2. Kohonen, T. *Self-organized formation of topologically correct feature maps*,
   Biological Cybernetics 43, 59-69 (1982). DOI: 10.1007/BF00337288.
   <https://doi.org/10.1007/BF00337288>
3. Salton, G.; Buckley, C. *Term-weighting approaches in automatic text retrieval*,
   Information Processing & Management 24(5), 513-523 (1988).
   DOI: 10.1016/0306-4573(88)90021-0.
   <https://www.sciencedirect.com/science/article/pii/0306457388900210>
4. Achlioptas, D. *Database-friendly random projections: Johnson-Lindenstrauss
   with binary coins*, Journal of Computer and System Sciences 66, 671-687 (2003).
   DOI: 10.1016/S0022-0000(03)00025-4.
   <https://cgi.di.uoa.gr/~optas/papers/jl.pdf>
