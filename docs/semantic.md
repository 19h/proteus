# Knowledge-grounded semantic retinas

[Documentation index](../README.md#documentation) · [Theory review](theory.md) · [Measured validation](validation.md#knowledge-grounded-implementation)

## Representation and scope

`proteus semantic train` implements the named backend
`knowledge-grounded-senses-events-v1`. It combines an explicit lexical graph,
corpus-adapted sense weights, local association coordinates and attributed event
bindings. It is a knowledge-grounded alternative to the reference SOM trainer.
Its semantic scores retain the graph and bindings instead of reconstructing
meaning from binary overlap.

```text
WordNet senses/relations ----+--> sense graph --> typed word scores
                            |
original text -> parser ----+--> occurrence contexts --> sense weights/examples
                            +--> local PPMI ----------> association scores
                            +--> bound events --------> support/conflict/unknown
                            |
                            +--> observed ancestor features --> sparse bitmaps

all resources, annotations and provenance --> compressed v5 retina
```

The lexical resource supplies knowledge absent from a single book. Corpus
training determines retained vocabulary, observed inflection-to-lemma links,
local associations, sense masses and source evidence. Default context-free
similarity takes the best lexical sense pair; its gains must not be attributed
to unsupervised corpus learning. Expected similarity and contextual queries
use the adapted priors. There is no learned contrastive neural semantic encoder,
calibrated confidence, general entailment engine or new scientific novelty claim.
The transformer is used for dependency annotation, not sentence embedding.

## Prepare, train and inspect

Preparation is explicit and optional. Native training and saved-model queries
require neither Python nor a network connection. The exercised environment uses
CPython 3.11 and the versions in [requirements-semantic.txt](../requirements-semantic.txt).
Install the parser model separately; no Rust query downloads it:

```sh
python3.11 -m venv .venv-semantic
.venv-semantic/bin/python -m pip install -r requirements-semantic.txt
.venv-semantic/bin/python -m spacy download en_core_web_trf

.venv-semantic/bin/python scripts/prepare_semantic.py lexicon \
    --nltk-data ./nltk_data --download --output wordnet.json
.venv-semantic/bin/python scripts/prepare_semantic.py annotate \
    --input corpus.txt --output corpus.jsonl

cargo +nightly build --release --locked
./target/release/proteus semantic train -i corpus.jsonl \
    --lexicon wordnet.json -o grounded.retina
./target/release/proteus semantic senses -r grounded.retina king
./target/release/proteus semantic similar -r grounded.retina king \
    --context 'the king rules his kingdom' -k 10
./target/release/proteus semantic similar -r grounded.retina king \
    --relation synonym -k 10
./target/release/proteus semantic score -r grounded.retina king queen
```

`--download` is needed only when the WordNet archive is absent. Omitting it uses
an existing archive. The export retains WordNet's license, version and archive
SHA-256. The parser defaults to `en_core_web_trf`, `--device cpu` and
`--batch-size 64`; `--device gpu` explicitly requires an available spaCy backend.
MPS preparation was exercised on the validation host. Rust's `gpu` feature is
independent of this Python preparation choice. `--model` selects another
installed pipeline, whose identity is recorded rather than silently substituted.
The smaller `en_core_web_sm` failed a basic active-voice probe during development;
a successful parse is an assumption to inspect, not a model-size guarantee.

Every parser record retains original paragraph text, its `source_start` byte
offset in the input, parser/model/backend identity, source SHA-256, and token
lemma, POS, dependency, head, sentence, morphology and UTF-8 byte offsets.
Hard wraps and CRLF inside paragraphs are preserved. Annotation files end with
a count and SHA-256 completion record. Rust rejects missing/mismatched completion,
including a valid prefix left by a failed producer. Preparation and retina
writing replace existing files atomically only after success. `-i -` supports
completed annotation streams on standard input.

The export schema is the public serde representation in
[corpus.rs](../src/semantic/corpus.rs) and [lexicon.rs](../src/semantic/lexicon.rs).
External annotators can produce those schemas without using spaCy. Supported
event patterns are specified below; dependency labels are not inferred from
unknown formats. Loading a different annotation policy is a model change.

## Word scores and sense evidence

All word queries use the saved normalization and retained corpus vocabulary.
Known phrases in the lexical source are counted over spans of at most five
parser tokens by default. Surface forms and their observed parser lemmas supply
candidate senses. Global entries retain all candidate senses, including senses
without confident corpus examples; no occurrence is presented as evidence for
an unobserved meaning. Unsupported lexical pairs return `null`, distinct from a
supported score of zero. No stopword filter or bitmap candidate cap changes
semantic ranking. Ties use lexical order.

| Relation | Dimensionless score / direction |
| --- | --- |
| `similarity` | Maximum sense-pair graph similarity without query context; with context, posterior-weighted per-query-sense maxima |
| `information-similarity` | Maximum sense-pair corpus-adapted information-content score; [mass contract](#information-content-similarity) |
| `expected-similarity` | Sum over query/candidate sense pairs weighted by their priors, or contextual query posterior |
| `synonym` | Shared explicit sense; context can weight the query sense |
| `antonym` | Explicit opposing sense edge, symmetrized at sense level |
| `hypernym` | Candidate is a strict ancestor of the query: `1 / (distance + 1)` |
| `hyponym` | Candidate is a strict descendant of the query: `1 / (distance + 1)` |
| `association` | Cosine of truncated, L2-normalized corpus PPMI rows; context/sense options rejected |

Strict relation queries return positive members only. Similarity, expected
similarity and association can include zero scores to fill the requested count.
Explicit `--sense ID` checks membership and conditions the query on that sense.
Candidate ambiguity remains explicit in the best-pair and expected contracts.

For senses `s,t`, graph similarity is 1 for identity, 0 for an explicit opposition
edge, and 0.9 for an explicit similar-to/verb-group edge. The constant 0.9 is a
declared heuristic, not a fitted probability. Otherwise different POS receive 0
(adjective satellites share adjective POS). Within a POS, let `d(s)` be maximum
root-to-sense path depth with roots at 1, and let `h` be the greatest depth of an
actual common ancestor. The score is `2h / (d(s) + d(t))`, with `h=0` for
disconnected components. No synthetic verb root is inserted. This is a specified
Wu-Palmer-style variant; it need not equal every library's Wu-Palmer score on a
multiply inherited graph. Ancestral edge distance uses the shortest retained path.

Glosses contain lexical definitions and examples. Treat each gloss and context
as token sets. Gloss IDF is `1 + ln((S + 1)/(df + 1))`, for `S` lexical senses.
The overlap `g(s,c)` is cosine between these IDF-weighted sets. For lexical prior
`p0(s)` (matching tagged lemma count plus one, normalized), occurrence weights are

```text
p(s | c) = softmax_s(ln p0(s) + g(s,c) / temperature)
p_word(s) = (prior_strength * p0(s) + sum_occurrences p(s | c))
            / (prior_strength + occurrence_count)
```

Query context uses `p_word` in place of `p0`. This is a gloss-overlap weighting
model, not calibrated word-sense disambiguation. Training contexts use sentence-
local parser lemmas; query context is normalized supplied text. Top examples
have original source byte ranges. Support counts deduplicate identical source
text by SHA-256; distinct text groups are not assumed statistically independent.
A posterior threshold is an evidence bookkeeping cutoff, not a confidence bound.

Association counting uses original parser-token distance within one sentence,
excluding the entire target phrase. Weight at distance `r` in window `W` is
`(W + 1 - r)/W`. If `C(w,c)` is accumulated mass, `R(w)` its row sum, and `M(c)`
the context marginal, coordinates are
`max(0, ln((C(w,c)/R(w)) / (M(c)^alpha / sum_j M(j)^alpha)))`.
Positive coordinates are sorted, truncated, then L2 normalized. This channel
keeps topical/substitution evidence separate from the lexical taxonomy.

## Target-aware contextual comparison

The original `senses` and `similar --context` interfaces retain their saved v1
interpretation. The explicit target-aware APIs add known-POS conditioning,
UTF-8 target exclusion, bilateral comparison and ambiguity diagnostics. They
reuse the stored gloss IDF and `sense_temperature`; no new parser or download
runs during inference.

```sh
proteus semantic disambiguate -r grounded.retina bank \
    --context 'bank river water shore' --pos noun \
    --target-start 0 --target-end 4
proteus semantic contextual-similar -r grounded.retina king \
    --context 'male sovereign ruler of a kingdom' --pos noun \
    --relation information-similarity -k 10
```

`ContextualWord` contains `word`, original `context`, optional `part_of_speech`
(`noun`, `verb`, `adjective`, `adverb`) and optional `target: {start,end}`. The
range uses original half-open UTF-8 byte offsets and must cover whole normalized
words matching the requested word or an observed lemma form. Optional `target_lemma`
is an explicit caller annotation for unobserved inflections; it requires a range
and must match the requested word/known lemma. The caller supplies that linguistic
annotation, just as it supplies POS; inference does not guess its correctness.
CLI `--target-lemma` exposes this choice. The exact range is
replaced by a space before scoring, preventing the target itself from supplying
gloss evidence. An absent range means the entire supplied text is surrounding
evidence; no occurrence is selected implicitly. CLI range endpoints require each
other. POS is supplied by the caller, not inferred, and adjective satellites
share adjective POS. Corpus-only vocabulary coverage remains explicit.

Candidate priors are conditioned and renormalized on POS, then passed through
the existing gloss-overlap equation. Results expose all source glosses, literal
matching terms, selected MAP sense, normalized masses, entropy in nats and the
largest-minus-second-largest mass. These are model-distribution diagnostics,
not calibrated linguistic confidence. Zero overlap returns the conditioned prior;
no lexical evidence returns a named unavailable reason.

`compare_contexts(left,right)` conditions both usages independently. It returns
MAP-sense equality, `sum_s p_left(s)*p_right(s)` for shared identities, and
`sum_(s,t) p_left(s)*p_right(t)*taxonomy_similarity(s,t)`. The scalar equations are
symmetric under argument reversal; f64 traversal rounding can differ. Even identical ambiguous contexts can have
posterior agreement below 1; agreement is not a calibrated equivalence score.

`find_similar_contextual` computes one query posterior for the entire ranking.
It averages the declared relation score conditioned on each query sense. For
similarity/information scoring, candidate senses use their maximum; expected
similarity additionally uses candidate priors. Candidate polysemy therefore
remains a limitation. Association is rejected because it has no sense-conditioned
contract. All retained entries are scored and ties remain lexical.

### Complete contextual evaluation

`semantic compare-contexts -r FILE --pairs JSONL` evaluates both the legacy gloss
baseline and target-aware MAP equality in one model load. Each line is:

```json
{"id":"bank-1","left":{"word":"bank","context":"bank money loan","part_of_speech":"noun","target":{"start":0,"end":4}},"right":{"word":"bank","context":"bank river shore","part_of_speech":"noun","target":{"start":0,"end":4}},"same_sense":false}
```

`same_sense` is optional. Unique IDs preserve separate repeated observations.
Malformed JSON, duplicate IDs and file errors abort; invalid target queries are
retained with error details and unavailable predictions. Reports include every
row, labelled/available counts, covered accuracy, accuracy counting unavailable
predictions as errors, common-coverage accuracies and paired improvements versus
regressions. Undefined accuracy is `null`. The CLI checks model/pair hashes
before and after execution.

[`evaluate_contexts.py`](../scripts/evaluate_contexts.py) converts an explicitly
supplied local WiC split and optional gold file into this schema. It records
original UTF-8 ranges, author-supplied target lemmas/POS, all identities, the command and native predictions, and
uses a paired bootstrap by target lemma. No split is downloaded or opened
implicitly. `--lexicon` audits normalized complete-context substrings against the
same lexical resource, rejecting mismatched resource hashes. The audit reports
neither/one-or-both/both-context overlap strata without excluding any observation.
Literal overlap identifies a resource-contamination risk, not a calibrated test
of memorization. Output publication is atomic and cannot replace an input.

```sh
python3 scripts/evaluate_contexts.py --retina grounded.retina \
    --data WiC/dev/dev.data.txt --gold WiC/dev/dev.gold.txt \
    --lexicon wordnet.json --output wic-dev.json
```

WiC examples can come directly from WordNet. An unseen pair-label split does not
make it resource-disjoint; contextual results must retain that limitation.
[The follow-up validation](validation.md#contextual-and-information-content-extension)
records development selection, source-overlap strata and frozen test evaluation.

## Information-content similarity

`--relation information-similarity` is an explicit Lin-style alternative to path
depth. It uses retained corpus entry counts and sense masses, plus one
pseudo-occurrence per lexical sense. Existing default scores and file semantics
remain unchanged. For canonical POS `p`, let `S_p` be its lexical senses and
`w(s) = 1 + sum_entries count(entry)*prior_entry(s)`. Then:

```text
N_p  = sum_(s in S_p) w(s)
M(a) = sum_(s with ancestor a, including a itself) w(s)
IC(a) = -ln(M(a) / N_p)
sim(s,t) = 2 * max_common_ancestor IC(a) / (IC(s) + IC(t))
```

Each distinct ancestor receives a sense's mass once, even with multiple DAG
paths. Ancestor probability is monotonic; a full POS root has IC 0. The lexical
pseudo-occurrences prevent zero-probability/unseen-subtree collapse. Count mass
includes retained phrases and is a modelled vocabulary-occurrence distribution,
not a count of independently sampled raw tokens. The fixed unit prior is part
of this named scorer, not selected by a parameter sweep.

Identity remains 1; explicit antonyms remain 0 and explicit similar-to/verb-group
edges retain the declared 0.9 override. Different POS or disconnected ancestry
receive 0. Distinct senses with zero denominator receive 0. POS-inconsistent
ancestry is rejected by this scorer. Thus these overrides and smoothing specify
a local variant, not a claim of exact equivalence to every implementation of Lin.

The information vector is derived lazily, cached immutably and omitted from
serialization. All required counts, priors and graph edges already reside in v5
models. Loaded and mapped queries agree without retraining or a format upgrade.
Source training reports retain their original algorithm identity. Word-pair
`semantic evaluate --relation information-similarity` declares the selected
contract in `score_kind`.

## Bound evidence search

```sh
.venv-semantic/bin/python scripts/prepare_semantic.py annotate \
    --text 'Dogs chase cats.' --output query.jsonl
proteus semantic search -r grounded.retina --query query.jsonl -k 10
proteus semantic events -r grounded.retina -k 20
```

Search requires exactly one complete annotated query document and at least one
supported event. It extracts verbal predicates, canonical active-subject and
active-object slots, indirect objects, complements and prepositional slots.
Passive subjects become active objects; passive by-objects become active
subjects. These are grammatical bindings, not universal thematic-agent labels.
Compound participant names, original offsets, determiner/numeric qualifiers,
finite tense, modals, adverbs and direct negation are retained. Binding identities
preserve one-letter identifiers and numbers even when vocabulary filters exclude
them. Evidence is matched by exact lemma or explicit shared lexical sense, with
an injective participant assignment and exact role/qualifier agreement.

For `Q` extracted query events, support count `P`, conflict count `N`, and explicit
penalty `lambda >= 0`, the directional score is `(P - lambda*N)/Q`.
An opposite-polarity matching assertion is a conflict; absent evidence is unknown.
Conflicts take precedence when a document contains both polarities. All source
documents participate, including documents with no extracted events. Returned
text and ranges allow inspection of the source assertion.

| Probe | Expected relation to `Dogs chase cats.` |
| --- | --- |
| `Cats are chased by dogs.` | Supported after voice normalization |
| `Cats chase dogs.` | Unknown: participant roles differ |
| `Dogs do not chase cats.` | Conflicted: matching assertion with opposite polarity |
| `Dogs may chase cats.` | Unknown: modal qualifiers differ |
| `Dogs chased cats.` | Unknown: tense qualifiers differ |

These states concern extracted assertions, not the truth of documents. Copular
predicates, coreference, ellipsis, coordination, reported speech and quantifier
logic are not general inference capabilities. A sentence-level negation attached
to another predicate makes scope unsupported; unsupported query scope is rejected.
The extractor bounds participant matching at 64 arguments. Lexical synonym
matching inside events is not full contextual disambiguation. Parsing errors,
polysemous substitutions and omitted relations can still cause errors; inspect
`events` and evaluate on the intended domain. Generic `text_similarity`, text
fingerprints, analogy and segmentation retain their unordered bitmap semantics;
use this explicit evidence API for role and polarity distinctions.

## Configuration, fingerprints and persistence

All resolved settings are persisted in `SemanticReport.config` and displayed by
`info`. Native training is deterministic for identical prepared inputs, binary
and platform; it does not have a random seed or SOM update count.

| Training option | Default | Effect |
| --- | --- | --- |
| `-d`, `--dimension` | 128 | Square non-spatial bit universe, `G = dimension^2` |
| `--min-count` | 2 | Minimum surface-word/phrase occurrence count |
| `--context-window` | 5 | Maximum original token distance |
| `--context-alpha` | 0.75 | Context marginal exponent in [0,1] |
| `--max-associations` | 32 | Positive PPMI coordinate cap |
| `--max-examples-per-sense` | 3 | Attributed example cap per sense |
| `--max-phrase-tokens` | 5 | Known-phrase span limit |
| `--lexical-prior-strength` | 5 | Prior pseudo-occurrence mass |
| `--min-sense-support` | 0.5 | Example/support bookkeeping threshold |
| `--sense-temperature` | 0.15 | Positive gloss-overlap temperature |
| `--sparsity` | 0.02 | Upper active-bit fraction |
| `--max-active-bits` | No extra cap | Additional absolute bit cap |

The library also accepts `SemanticConfig.text` and fingerprint pooling settings;
the CLI uses the documented tokenizer defaults. Frequencies divide each entry's
count by total retained word-plus-phrase occurrences, not raw token count.
Fingerprint features sum `sense_prior * ancestor_depth/(distance + 1)` over
observed lexical ancestors. SHA-256 of ancestor identity chooses the bit slot;
collisions are merged and strongest slots retained. No random padding supplies
missing evidence. Unknown lexical entries keep empty bitmaps. Spatial lookup
rejects these models. Bitmap similarity remains available as a distinct API;
ordinary word ranking uses the retained semantic graph, including mapped reads.

Semantic retinas use format v5: the existing 64-byte header and indexed sections,
with a metadata envelope consisting of an eight-byte little-endian decoded JSON
length and a zlib stream. Readers enforce an exact decoded length at most 1 GiB,
reject trailing compressed bytes, and validate graph, source ranges, report
hashes, vocabulary, configuration and re-extracted events. Mapped reads still
materialize rich metadata and the graph; they only avoid retaining all decoded
bitmaps. Metadata is not an unbounded out-of-core source store.

Legacy readers v1-v4 remain available; nonsemantic writers still emit v4. Old
binaries reject v5 instead of interpreting graph features as an ordinary retina.
Migration preserves semantic scores, original annotations, graph and hashes.
A model carries complete source text and the lexical resource; its size and
publication contents must be assessed from those actual stored resources.

## Costs, assumptions and falsification

Let `T` be parser tokens, `L` maximum phrase span, `W` context window, `O` retained
occurrences, `B` maximum candidate senses, `J` context/gloss token-set size,
`S` lexical senses, `E` lexical edges, `A` maximum retained ancestors, `C` observed
association pairs, `V` vocabulary, and `F` source events. Ordered maps add
logarithmic lookup factors. Phrase-span tokenization costs up to `O(T L^2)` token
work; corpus accumulation costs approximately `O(O(W + BJ))` plus ordered-map
operations. Ancestor construction is bounded by `O(E A log A)`, using `O(S A)`
space, with explicit depth 256 / ancestor 4096 limits. Fingerprint construction
adds up to `O(VBA log(BA))` ordered accumulation/ranking work and
`O(V min(G,BA))` stored bit positions. Glosses, source annotations, associations
and sense/event evidence add `O(SJ + T + C + OB + F)` space. Parser inference is
additional work governed by the selected external model, not these Rust bounds.

Information-content construction adds `O((S + VB) A)` mass accumulation and
`O(S)` cache storage on first use; ancestor sets are already indexed. Subsequent
information pair scores have the same ancestor-intersection order as depth scores.
Target-aware disambiguation costs `O(BJ)` set work plus tokenization and
`O(B log B)` sorting; bilateral comparison adds `O(B^2 A log A)`. Contextual
ranking computes the posterior once, then scans all candidates; it is exhaustive.
Context-pair evaluation retains `O(PB + input_bytes)` state for `P` pairs.

One pair query costs up to `O(B^2 A log A)` graph lookup work, plus contextual
overlap when requested. Exact ranking scores every vocabulary entry and sorts
`O(V log V)` results; there is no sublinear latency claim. Evidence search scans
source/query event pairs. For `R <= 64` participants, building lexical edges
costs up to `O(R^2 B^2)` sense-membership work, followed by augmenting paths in
`O(R^3)` time and `O(R^2)` space. Ranking materializes per-document evidence and
copies source text before truncation; it is not a streaming top-k operation.
Saving/loading validates all retained metadata. Compression reduces disk size, not asymptotic resident memory.

| ID | Assumption / dependent result | Stress or falsification probe |
| --- | --- | --- |
| S1 | Lexical relations are relevant to the intended word task; lexical quality depends on this. | Matched-coverage pair evaluation; stratify POS/domain; compare the corpus association channel and prior-weighted ablation. |
| S2 | Gloss overlap selects useful senses; contextual results depend on this. | Opposed bank contexts and explicit sense controls; WiC is additionally evaluated with lexical-source overlap audited; resource-disjoint contextual accuracy remains unknown. |
| S3 | Parser bindings preserve the intended assertion; evidence scores depend on this. | Real-parser active/passive, role reversal, polarity, modality and tense probes; audit original spans; reject unsupported scope. |
| S4 | Best-sense comparison is useful despite polysemy; context-free ranking depends on this. | Compare expected similarity and contextual rankings; inspect rare-sense false matches and relation-specific neighbors. |
| S5 | Prepared inputs identify complete resources; reproducibility depends on this. | Completion checksum, parser/resource hashes, byte-identical retraining, save/load/migration/mapped agreement and malformed metadata tests. |
| S6 | The in-memory representation fits the workload; capacity claims depend on this. | Measure process memory and artifact sizes; enforce decoded metadata bounds; no corpus-size extrapolation. |

The extension additionally records these assumptions and their dependent results:

| ID | Assumption | Basis | Dependent result | Stress test | Falsification probe | Status |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | Smoothed subtree mass is useful commonality for the target lexical task. | Retained corpus counts, adapted priors and monotonic ancestor probability. | Information-content rankings. | Rare/unseen senses and multiple ancestry paths. | Independent subtree-mass oracle and matched-coverage pair evaluation, including relatedness losses. | Numerical fixtures pass; task-specific empirical evidence is recorded separately. |
| C2 | Supplied target POS, byte range and optional lemma are correct. | Explicit caller or dataset annotations. | Contextual inference. | Wrong POS, UTF-8 cuts, partial words and unobserved inflections. | Reject invalid ranges/identity; compare POS/target-conditioned inference on complete observations. | Validated for exercised schemas/fixtures; annotation correctness remains caller-dependent. |
| C3 | Unseen pair labels measure useful contextual discrimination despite shared lexical resources. | Explicit train/dev/test boundaries, with WordNet-derived source examples. | Interpretation of WiC results. | Full-context source overlap and missing corpus words. | Report overlap strata, OOV/error denominators, paired changes and lemma-cluster intervals. | Pair-label holdout does not establish resource-disjoint generalization; that remains unknown. |
| C4 | New scoring can reuse v5 evidence without altering old scores or bytes. | Derived immutable caches omitted from serialization; explicit query methods. | Compatibility and reproducibility. | Cold/warm caches, loaded/mapped readers and save/reload. | Compare saved bytes and archived default predictions. | Fixture coverage passes; runtime identity is included in the validation record. |

| Impact | Adjacent opportunity or limit | Bound |
| --- | --- | --- |
| High | Structured evidence distinguishes some meanings provably collapsed by unordered pooling. | Verified on the supported probes; not general natural-language inference. |
| High | External lexical knowledge materially changes the available signal. | Report the resource and matched coverage; gains are not proof of corpus-only learning. |
| High | Best-sense scores can promote an irrelevant rare sense. | Expose senses and context; expected scoring is an explicit ablation, not an automatic cure. |
| Medium | Complete evidence and graph improve auditability but increase load cost. | Compressed bounded metadata; sharded source storage and graph indexes remain future work. |
| Medium | Domain-specific sense supervision could improve contextual weighting. | Contrastive training, calibration and domain holdouts remain research directions. |

[Contract tests](../tests/semantic_contract.rs) check independent numerical
fixtures, roles/polarity/scope, phrases, missing evidence, deterministic files,
loaded/mapped migration, CLI wiring, malformed resources and atomic failure.
[Preparation tests](../tests/test_prepare_semantic.py) check offsets, hashing and
atomic publication. The [validation record](validation.md#knowledge-grounded-implementation)
separates those mechanics from corpus measurements and their limitations.

## Primary sources

- [Princeton WordNet](https://wordnet.princeton.edu/) and its [license](https://wordnet.princeton.edu/license-and-commercial-use): explicit English lexical evidence, retained in exports.
- [NLTK WordNet interface](https://www.nltk.org/howto/wordnet.html): synsets, lemma counts and lexical relation access.
- [Wu and Palmer, 1994](https://aclanthology.org/P94-1019/), DOI [10.3115/981732.981751](https://doi.org/10.3115/981732.981751): taxonomy-based similarity predecessor; the exact local variant is specified above.
- [Lin, 1998, An Information-Theoretic Definition of Similarity](https://www.cs.swarthmore.edu/~richardw/classes/cs65/f08/litreview/phyo.pdf): information/commonality predecessor; local mass estimation and overrides are specified above.
- [WiC author dataset and README](https://pilehvar.github.io/wic/): same-meaning task, target indices, explicit labels and source limitations.
- [spaCy English pipelines](https://spacy.io/models/en) and [model installation](https://spacy.io/usage/models): explicit dependency annotation resources.

No source establishes the empirical quality of this implementation; that requires
the local experiments and domain validation reported separately.
