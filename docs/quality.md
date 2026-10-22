# Retina quality and evaluation

[Documentation index](../README.md#documentation) | [Algorithm](algorithm.md)

Deterministic training and exact retrieval do not establish semantic accuracy.
The quality pipeline separates the evidence used to construct fingerprints from
the labelled pairs used to evaluate their rankings. See the
[measured results and limits](validation.md#final-distributional-backend-and-broader-corpus-comparison).

## Spatial evidence folding

The online SOM still trains on snippets and assigns every usable snippet to the
frozen map. Let `h(w, j)` be the number of snippets containing word `w` assigned
to cell `j`, and let `n(j)` be the number of all snippets assigned to that cell.
Each snippet contributes at most one incidence per word.

For a Gaussian standard deviation `sigma > 0`, define the finite kernel:

```text
radius = ceil(3 * sigma)
g(i, j) = exp(-distance_grid(i, j)^2 / (2 * sigma^2))
         when both axis distances are <= radius, otherwise 0
K(i, j) = g(i, j) / sum_i g(i, j)

word_evidence(w, i) = sum_j h(w, j) * K(i, j)
background(i)       = sum_j n(j) * K(i, j)
score(w, i)         = word_evidence(w, i) / background(i)^beta
```

Coordinates and `sigma` use cell units. Counts, kernel weights and scores are
dimensionless. The kernel enumerates each toroidal destination exactly once,
including the antipode of an even-sized grid. Source normalization preserves
mass at planar boundaries as well. Cells with no word evidence are excluded;
background is positive wherever word evidence is positive.

Retain the largest positive scores up to the existing density/cardinality cap.
Equal scores use the smallest cell index. The optional Gaussian preset is
`sigma = 1` cell and `beta = 0.5`; raw incidence (`sigma = 0`) remains the default.
Local interpolation lets nearby context assignments share bits;
background correction reduces concentration at densely occupied cells. This is
an explicit engineering extension, not a claim to reproduce Cortical.io's
proprietary implementation. Bits represent interpolated evidence, not additional
observations. The finite support and cardinality cap are separate operations.

`--folding-sigma 0` selects the original raw-incidence algorithm, including its
observed-cell-only support and count ordering. It ignores background correction.
Metadata without folding settings is interpreted as raw incidence; loading an
old model never silently changes its bits or query results.

## Optional local association features

`--feature-model context-ppmi` replaces lexical snippet coordinates with a
TF-IDF weighted composition of word-context vectors. The default remains
`--feature-model tfidf`. Context PPMI is an alternative feature representation;
it is not skip-gram, an SVD factorization, or an independently established
improvement on every corpus.

For target word `w` and context feature `c`, count occurrences up to
`--context-window W` original token positions away, with weight
`(W + 1 - distance) / W`. Exclude the center position and never cross a snippet
boundary. Vocabulary filtering preserves distances rather than bringing words
closer together. Let these weighted counts be `C(w, c)`:

```text
p(c | w) = C(w, c) / sum_c C(w, c)
context_mass(c) = sum_w C(w, c)
p_alpha(c) = context_mass(c)^alpha / sum_d context_mass(d)^alpha
PPMI(w, c) = max(0, ln(p(c | w) / p_alpha(c)))
```

More explicitly, `p_alpha(c) = (sum_w C(w,c))^alpha /
sum_d (sum_w C(w,d))^alpha`, with default `alpha = 0.75`.
Only observed pairs are evaluated. Empty rows remain zero; no arbitrary vector
is substituted. Apply the optional seeded projection to each PPMI row, then
normalize it. Compose these vectors using the existing sublinear TF-IDF weights
for feature-eligible words, then normalize the snippet. Zero snippets are omitted
and counted exactly as in lexical training.

The association and context-distribution smoothing are motivated by
[Levy, Goldberg and Dagan (2015), DOI 10.1162/tacl_a_00134](https://aclanthology.org/Q15-1016/).
The particular normalized composition and subsequent SOM are implementation
choices here, not a reproduction of that paper's evaluated models.

## Spectral projection

`--projection-method spectral --projection-dim D` on the context trainer, and
`--embedding-dimension D` on the distributional trainer, use orthogonal subspace
iteration on the unnormalized positive-PMI matrix `A`. With seeded Rademacher
initial coordinates and column-orthonormalization `orth`:

```text
Q = orth(random feature-by-D matrix)
repeat power_iterations times:
    U = orth(A * Q)
    Q = orth(transpose(A) * U)
word_coordinates = A * Q
if whitening_enabled:
    word_coordinates = orth(word_coordinates)
```

The default is two iterations. The distributional command additionally whitens
the resulting columns before normalizing word rows; `--no-whitening` retains
their original scale. This equalizes spectral directions instead of allowing
the largest singular directions to dominate cosine. Dimensions exceeding the
feature count are
clamped to that count and the effective dimension is recorded. Multiplication
accumulates each coordinate in fixed binary64 order, then stores binary32.
Orthogonalization uses two modified Gram-Schmidt passes and drops residual
columns at or below `1e-6` of their initial norm. Rank-deficient columns stay
zero. This is a randomized approximation to the dominant right singular
subspace, not an exact SVD or a fixed approximation-error guarantee. See
[Halko, Martinsson and Tropp (2011), DOI 10.1137/090771806](https://doi.org/10.1137/090771806).

## Non-spatial distributional retinas

```sh
proteus train-distributional -i corpus.txt -o distributional.retina \
    --embedding-dimension 128 --power-iterations 2 --seed 42
```

This separately named backend constructs the local PPMI matrix and its spectral
word vectors directly. It does not train a SOM and does not accept an iteration
count for one. Zero word vectors are omitted. For each nonzero word vector `x`,
it uses a shared collection of `G = dimension^2` independent Gaussian vectors
`g(i)` and retains the `K` largest responses:

```text
g(i, j) ~ Normal(0, 1)
response(x, i) = dot(x, g(i))
K = min(ceil(sparsity * G), max_active_bits, G)
fingerprint(x) = positions of the K largest responses
```

Equal responses use ascending feature index. Every encoded word has exactly
`K` bits. These bits identify synthetic angular features, not observed snippets.
They are not padding to a minimum incidence count. Spectral initialization and
Gaussian directions use distinct ChaCha8 streams derived from the saved seed.

The real-valued responses have `E[response(x,i) response(y,i)] = dot(x,y)`:
expand the two sums and use `E[g(i,j)g(i,k)] = 1` when `j=k`, otherwise zero.
For unit vectors their joint Gaussian correlation is their cosine. The binary
rank cutoff is an explicit lossy encoding of those responses. It is not an
unbiased cosine estimator or a finite-size distortion guarantee. Quality is
measured on the final bitmaps, not inferred from the response identity.

Bit positions have no learned two-dimensional adjacency. Grid layout is a
storage/display convention. `Retina::has_spatial_topology()` returns false,
and the CLI rejects `semantic-lookup` for these models. Exact `similar`, bitmap
`lookup`, composition, segmentation and fingerprint images remain available;
image proximity is not semantic proximity. The reference `train` command keeps
its context-map contract and default behavior.

Older version-1 angular reports without a whitening field retain `whiten=false`
when loaded; their bits are never recomputed. The stored `AngularReport` is
separate from `TrainingReport`: it contains the
actual feature dimensions, seed, corpus digest, vocabulary and omitted-zero-word
count, without inventing SOM updates or quantization errors. The algorithm ID is
`distributional-ppmi-spectral-angular-v2`. A word vector averages its contexts;
distinct senses and synonym-versus-antonym relations remain limitations.

## Evaluate a model

Supply UTF-8 tab-separated `word1`, `word2`, and finite reference score columns.
Additional columns are allowed. Headers are skipped only when requested:

```sh
proteus evaluate -r model.retina --pairs pairs.tsv > evaluation.json

# SimLex-999 has a header and its reference score is column 4.
proteus evaluate -r model.retina --pairs SimLex-999.txt \
    --score-column 4 --header > evaluation.json

python3 scripts/compare_quality.py baseline.json candidate.json --split dev
python3 scripts/compare_quality.py baseline.json candidate.json --split test
```

The evaluator uses the stored tokenizer and ordinary binary cosine. CLI output
also records SHA-256 hashes of both input files and rejects a detected change
during evaluation. It reports all predictions, missing words explicitly as JSON `null`, evaluated-pair count,
coverage, zero-similarity count and Spearman correlation with average ranks for
ties. Constant predictions or fewer than two covered pairs have an undefined
correlation (`null`), not a successful score of zero. Out-of-vocabulary pairs are
never assigned artificial zero similarities.

Comparison uses only pairs covered by **both** models. Reference observations and
labels must match in order. Repeated pairs remain separate rated observations,
including the duplicates in the original WordSim-353 distribution. Development
membership
is fixed by `SHA256(UTF8(word1 + TAB + word2)) mod 3 == 0`; test membership is its
complement. Select settings using development results before examining the test
split. Paired bootstrap intervals use 2,000 resamples and seed 42 by default.
They describe pair resampling; repeated words violate pair independence, so the
interval does not prove cross-domain generalization.

[SimLex-999's authors](https://fh295.github.io/simlex.html) explicitly distinguish
word similarity from topical association. A one-book corpus may provide little
support for many pairs. Coverage and low absolute correlations must accompany
comparisons rather than being hidden behind relative improvements.

## Support diagnostics

Training reports now include the number of words supported by only one usable
snippet, duplicate bit-pattern count, mean active bits, largest cell's fraction
of mapped snippets, and effective occupied cells `exp(-sum p(i) ln p(i))`.
These describe support and compression; they are not semantic-accuracy metrics.
Old metadata reports them as unavailable. All settings and diagnostics survive
saving and migration. Both normal and memory-mapped readers recover the same
fingerprints. Angular reports describe encoding statistics separately.

## Assumptions, limits and checks

| ID | Assumption | Basis | Dependent result | Stress test | Falsification probe | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | Nearby learned cells carry related context evidence. | SOM neighborhood objective. | Spatial folding quality. | Sparse words and congested cells. | Fixed raw-versus-smoothed labelled pairs. | Not established by the first held-out trial. |
| Q2 | Occupancy correction improves word similarity. | Density varies between map cells. | Gaussian preset. | Common versus rare words. | Development-selected settings on held-out pairs. | Initial held-out gain falsified; preset stays optional. |
| Q3 | Local contexts contain useful word-similarity signal. | Distributional representation and independent PPMI fixture. | PPMI/spectral modes. | Substitution without co-occurrence; sparse rows. | Compare full pipeline against lexical baseline. | Improved final bitmap rankings on WikiText-2 across three seeds; Moby-Dick remains weak. |
| Q4 | Reference pairs represent an intended application. | Benchmark task definitions. | Transfer of quality measurements. | Missing and rare vocabulary. | Report common coverage; use a separate domain benchmark. | General-domain transfer remains unknown. |
| Q5 | Reproducibility is scoped to one binary and numerical platform. | Canonical accumulation and seeded streams. | Byte identity and repeatable queries. | Fresh processes and both readers. | Compare complete files and output lists. | Confirmed for exercised fixtures only. |
| Q6 | Sparse angular response ranks retain enough word-vector geometry. | Shared isotropic Gaussian responses. | Non-spatial word quality. | Identical, orthogonal and opposite vectors; multiple seeds. | Compare final bitmap scores on held-out labelled pairs and matched coverage. | Numerical invariants verified; empirical scope is recorded in validation. |

| Impact | Additional pattern or limit | Bound |
| --- | --- | --- |
| High | Interpolation can merge senses when the map geometry is poor. | Keep raw incidence available and evaluate instead of assuming smoother means better. |
| High | Sparse corpus support and corpus domain constrain similarity. | Report single-context words, pair coverage and absolute correlation. |
| Medium | Narrow projections can lose distinctions even on simple corpora. | Compare exact coordinates and projected models; deterministic output does not guarantee quality. |
| Medium | PPMI requires additional word-context storage. | For N tokens, window W, C observed pairs and D coordinates, additional work is O(NW + CD + ND) with projection and memory O(C + VD). |
| High | Angular bits have no spatial neighborhood meaning. | The separate command and metadata expose this; spatial lookup rejects it. |
| Medium | Spectral reduction adds O(qCD + q(V+F)D^2) work and O(C + (V+F)D) storage for q power iterations. | Low-rank approximation is explicit; zero-rank columns stay zero. |
| Medium | Angular encoding costs O(VGD) after word vectors, with O(GD) shared directions and O(G) per-worker scores. | G is the bit universe, D the vector dimension, and V the encoded vocabulary. |
| Medium | A large interpolation radius increases folding cost. | For H kernel offsets and I distinct word/cell incidences, aggregation is O((G + I)H), plus candidate sorting; shared state uses O(G + H) memory. |

Quality gates separate independently calculated numerical fixtures, corpus
quality measurement, deterministic persistence, legacy compatibility and style.
The numerical tests cover planar mass, toroidal wraparound without duplicate
updates, bounded support, zero-sigma compatibility, PPMI substitution without
co-occurrence, forbidden cross-snippet contexts, tied-rank correlation and
undefined correlation. Integration tests cover both feature encoders, exact and
projected coordinates, invalid settings, repeated files and both query readers.
