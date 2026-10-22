# Semantic theory review and redesign

[Documentation index](../README.md#documentation) | [Current algorithms](algorithm.md) |
[Quality methods](quality.md) | [Experiment record](validation.md#theory-audit-and-independent-verb-evaluation)

The review below records the pre-implementation analysis at `afebd3a`. Its
counterexamples and historical measurements are preserved. The subsequent
[knowledge-grounded implementation](semantic.md) now supplies explicit senses,
typed relations and bound source evidence; [new validation](validation.md#knowledge-grounded-implementation)
records the implementation results. Proposed contrastive learning and calibrated
inference are still unimplemented. References to "current" encoders below refer
to the reviewed revision unless explicitly stated otherwise.

## Conclusion and scope

Semantic Folding supplies a representation architecture, not a sufficient
learning objective for semantic equivalence. Current Proteus representations
discard distinctions before binary encoding: contexts omit argument direction,
and word-type representations pool senses. Fresh verb results are already weak
at the raw-context stage. Increasing SOM updates, grid size, or bit count cannot
recover distinctions already removed there.

The proposed next architecture learns **sense- and relation-specific
representations**, preserves structured evidence, and uses sparse codes for
candidate retrieval. It measures semantic quality before optimizing compression.
This is a research design, not an implemented replacement or a demonstrated
breakthrough. Its components have published predecessors; novelty and the size
of any resulting improvement are unknown.

This review concerns lexical relevance, contextual meaning and compositional
retrieval. It does not evaluate the biological validity of HTM or claim that a
language representation is a model of cortical computation. The experiments
below use source revision `afebd3a`; production training and file formats are
unchanged by this review.

## What the original theory establishes

Webber's [Semantic Folding white paper, version 1.2](https://arxiv.org/abs/1511.08855)
describes corpus snippets, their placement on a two-dimensional map, and word
incidence fingerprints. Its numerical mapping procedure is not specified enough
to uniquely reproduce a trainer. Proteus's online SOM is an explicit engineering
choice. The paper also distinguishes association from synonymy; it does not
warrant treating every high-overlap result as a synonym.

| Claim or construction | Status for this project |
| --- | --- |
| A sparse set can support fast overlap operations. | A representation property; measured runtime remains workload-dependent. |
| Corpus contexts contain information about word usage. | Supported in some tasks; not sufficient for every semantic relation. |
| A two-dimensional arrangement grounds linguistic meaning. | Not established by map construction or biological analogy. |
| Union retains the information of its constituent representations. | Retains activated positions before truncation, but loses membership and binding. |
| A metric on bit patterns establishes a metric on meanings. | Requires additional assumptions about representation and semantic identity. |

The last three rows are distinctions between a data structure and what can be
inferred from it. They can be tested without making a judgment about the entire
theory.

## Where information is lost

```text
documents and occurrences
    |
    +--> tokenization and phrase boundaries
    |
    +--> undirected context counts     loses argument direction
    |
    +--> one row per word              merges senses and grammatical uses
    |
    +--> spectral projection           removes some context distinctions
    |
    +--> binary encoding               quantizes retained geometry
    |
    +--> text pooling                  removes constituent order and binding
```

These losses are different. A successful numerical test of the last two stages
cannot establish that the first stages supplied the required meaning.

### 1. Lost distinctions cannot be learned downstream

Let `T(x)` be a representation of input `x`, and let the final encoder be
`F(T(x))`. If `T(x) = T(y)`, then every deterministic downstream encoder gives
`F(T(x)) = F(T(y))`. If a task requires different answers for `x` and `y`, that
representation is insufficient for the task. Randomness independent of the
distinction supplies no evidence for a correct distinction either.

For lexical TF-IDF, reversing a snippet preserves all term counts. For the
current PPMI encoder, reversing a complete snippet preserves every pair's
absolute token distance, hence its undirected weighted count. With unchanged
snippets, vocabulary and seed, later training cannot recover argument direction.
Floating-point accumulation could introduce incidental differences on other
inputs; such differences would not encode a learned grammatical distinction.

The controlled audit reversed all eight three-word documents, including
`dogs chase cats` and `doctors treat patients`. **Both trainers produced identical
word fingerprints for all 23 vocabulary entries.** Their file hashes differ
because corpus provenance differs. Both also produced the same nonempty text
fingerprint for `dogs chase cats` and `cats chase dogs`.

Reproduce with the standard-library-only audit:

```sh
python3 scripts/audit_semantic_invariants.py \
    --binary target/release/proteus \
    --output target/validation/theory-reproduction
```

The output directory must be new. A successful run confirms the documented
counterexamples, not semantic accuracy.

### 2. Exact overlap does not use spatial adjacency

For a permutation matrix `P` and nonzero binary vectors `a`, `b`:

```text
cos(Pa, Pb) = (a^T P^T P b) / (||Pa|| ||Pb||) = cos(a, b)
```

An arbitrary shared permutation destroys the displayed map's neighborhood
layout while leaving every ordinary cosine query unchanged. The map can affect
how codes are learned, but its displayed adjacency is not consulted by that
score. Spatial interpolation and spatial lookup are different algorithms.

Consequently, two-dimensional appearance is not an acceptance criterion for
semantic retrieval. Learn the retrieval representation first; derive a display
layout separately if useful. This does not prove that every spatial regularizer
is ineffective.

### 3. Union is not reversible composition

`{a} union {b} = {a, b}`. The result cannot identify whether the input contained
two constituents or one constituent with both features. It also cannot recover
their order. Weighted word pooling remains commutative when weights depend on
word counts rather than argument roles.

Under an illustrative independent-uniform-set model, with `G` positions and
`K` positions per constituent, the expected union size after `m` constituents is:

```text
E[union_size] = G * (1 - (1 - K/G)^m)
```

For `G = 16384`, `K = 328` and `m = 50`, approximately 64% of positions are
occupied before a cap is applied. This is a null-model calculation, not a
measurement of correlated words. Keeping only 328 positions then discards
activated positions in addition to the binding information already lost.

### 4. A distance on codes does not validate semantic identity

Distinct words can have identical codes. Even a valid metric on codes therefore
does not prove that distance zero means identical meaning.

For unequal cardinalities, `1 - cosine` is not generally a metric: with sets
`A = {0}`, `B = {0, 1}`, `C = {1}`, the direct distance is `1`, while the sum
through `B` is approximately `0.585786`. For **fixed positive** cardinality `K`, binary
`1 - cosine = |A symmetric_difference B| / (2K)` is scaled Hamming distance;
the same counterexample does not apply. Neither case identifies synonymy by
itself.

### 5. Whitening is a modeling choice

For full-column-rank coordinates `Y`, exact whitening has row Gram matrix:

```text
Z = Y * (Y^T Y)^(-1/2)
Z Z^T = Y * (Y^T Y)^(-1) * Y^T
```

This is a projector onto the retained column space. It does not preserve the
original cosine geometry. If `Y` is square and invertible, the projector is the
identity: all different rows become orthogonal. For rows `(3, 1)` and `(1, 2)`,
cosine changes from `0.707107` to zero under exact full-rank whitening. The audit
checks the projector independently in binary64 arithmetic.

Current truncated, approximate whitening is not this square limiting case.
The counterexample invalidates a universal quality argument for whitening;
it does not invalidate the recorded empirical gain on some data. Regularized
whitening or singular-value weighting needs separate evaluation.

### 6. Binary overlap has a null background

For two independent uniformly chosen `K`-element subsets of `G` positions,
intersection size is hypergeometric. Therefore:

```text
E[intersection] = K^2 / G
E[binary_cosine] = K / G
Var(intersection) = K * (K/G) * (1 - K/G) * (G-K)/(G-1)
```

At `G = 16384`, `K = 328`, expected overlap is `6.5664` positions, expected
cosine is `0.020020`, and its standard deviation is `0.0076563`. All are
dimensionless except the count of positions. Learned word codes are correlated
and nonuniform; these values are not calibrated significance thresholds for
their similarities.

## What the fresh experiment changes

The official [SimVerb-3500](https://aclanthology.org/D16-1235/) test split contains
3000 verb pairs. Six previously trained WikiText-2 models were frozen before
this benchmark was downloaded: reference SOM and distributional models at seeds
1, 42 and 123. No model parameters were selected from these results.

All rows below use the same **2381 covered pairs out of 3000**, or **79.4%**.
Spearman correlations are dimensionless. Raw PPMI has no random seed.

| Seed | Reference SOM | Raw PPMI | Dense spectral, whitened | Final angular bitmap |
| --- | ---: | ---: | ---: | ---: |
| 1 | -0.009760 | -0.002096 | 0.008803 | 0.003263 |
| 42 | 0.006532 | -0.002096 | 0.002636 | -0.013573 |
| 123 | 0.001478 | -0.002096 | 0.021957 | 0.009643 |

The dense vectors were recovered using the current feature code, unchanged
except for an observation callback before projection. Normalized corpus hashes
matched, and re-encoding 65 deterministic vocabulary positions per seed
reproduced their stored bitmaps exactly. This is evidence of a matched diagnostic
path, not an independent numerical oracle for the feature implementation.

The final bitmap retains substantial dense ranking information on these pairs
(dense/bitmap rank correlation `0.882` to `0.892`), but neither representation
ranks semantic similarity usefully on this test. **The observed failure is
already present in the raw context representation.** These measurements do not
separate corpus insufficiency from the limitations of undirected contexts or
word-type aggregation; that requires further controlled comparisons.

At every seed, the average bitmap similarity of the 86 covered antonym pairs
exceeds that of the 213 covered synonym pairs. This is a descriptive diagnostic,
not a matched-frequency causal test. Raising the minimum evaluation frequency
does not resolve the seed-42 failure; the detailed record retains denominators.

The dataset is newly evaluated here, but 142 of its 3000 unordered pairs overlap
the previously inspected SimLex dataset. Excluding these as a sensitivity check
leaves 2244 commonly covered pairs; bitmap correlations remain between `-0.006827`
and `0.019833`. The primary result retains all 3000 rows. SimVerb is now an
inspected benchmark and cannot be reused as an untouched test of later designs.

The earlier SimLex/WordSim gains remain valid for their recorded scope. They
cannot support a claim of general semantic competence. Complete hashes,
predictions and reproduction details are in the linked experiment record.

## Proposed architecture

The highest-priority change is the meaning represented, followed by how it is
compressed. Four separable changes address different failures.

```text
raw documents + offsets + source identities
                  |
       phrase and occurrence analysis
                  |
     sense hypotheses + typed relations
                  |
      +-----------+-------------+----------------+
      |                         |                |
 substitution/equivalence    association     directed relations
      |                         |                |
      +------- learned representations ----------+
                  |
         sparse candidate indexes
                  |
 query + context + requested relation
                  |
     candidate retrieval -> evidence-aware reranking
                  |
       ranked results + support + ambiguity
```

### A. Separate the semantic questions

| Query relation | Intended evidence | Illustration for the royal-title sense of `king` |
| --- | --- | --- |
| Substitution/equivalence | Preserves meaning in a specified sentence | `monarch`, where the sentence permits it |
| Shared function/type | Similar grammatical or conceptual role | Another kind of ruler |
| Association | Common events, topics or entities | `throne`, names of particular rulers |
| Taxonomic relation | Directed type assertion | `king -> ruler` |
| Opposition | A specified contrast relation | Requires its own labeled relation; no generic opposite is assumed |

These illustrations define questions, not a manually curated retrieval answer.
For example, `queen` is not interchangeable with `king` in every sentence.

[Turney's dual-space model](https://arxiv.org/abs/1309.4035) already separates
domain and function. [Dependency-based embeddings](https://aclanthology.org/P14-2050/)
provide a tested predecessor for typed contexts. Neither establishes that syntax
alone distinguishes synonyms from antonyms. The proposed extension keeps
relation-specific outputs and learns their ranking objectives explicitly.

A symmetric similarity between the same word representations cannot express
the direction of `is-a`, causation or entailment. Directed relations need separate
query and candidate representations, or a directed scorer. They must not inherit
a symmetric-cosine contract.

### B. Represent occurrences and senses before pooling word types

Represent a lexical entry as a bounded collection of sense hypotheses:

```text
entry(w) = {sense_id, prior_mass, representation_by_relation,
            independent_support, source_references} for each sense
```

Use local context to estimate `p(sense | word, query_context)`. For a context-free
word query, expose sense groups or use an explicitly declared prior mixture.
Do not select the maximum over arbitrary sense pairs without checking hubness:
words with many senses would receive extra opportunities for accidental matches.

A concrete symmetric substitution score is:

```text
score(w, c; v, c') = sum_s sum_t p(s | w,c) * p(t | v,c')
                                    * kernel_substitute(z[w,s], z[v,t])
```

This equation is a proposed scoring contract, not a calibrated probability of
equivalence. Context-free queries replace the posterior with a declared prior;
different sense frequencies can then legitimately change results.

Preserve multiword units and offsets: `King-Post` should be available as a phrase
hypothesis rather than only two independent word tokens. A parser or phrase
extractor may make mistakes, so retain the original text and competing analyses.

[Reisinger and Mooney](https://aclanthology.org/N10-1013/) established
multi-prototype word representations; [context2vec](https://aclanthology.org/K16-1006/)
learns context representations for substitution and disambiguation.
[Wang et al. (2025)](https://arxiv.org/abs/2504.06036) also construct sense
dictionaries from contextual embeddings and distill smaller models. Reusing these
ideas is not a new scientific contribution by itself.

### C. Train on equivalence and contrast, not proximity alone

Construct positive and negative examples for a declared relation. For
substitution, include same-topic negatives, antonyms, participant-role swaps and
entity/type confusions. Positives must preserve the intended sense. A masked
language model can propose candidates, but predicting a plausible completion
does not establish equivalence: both `hot` and `cold` can complete an
underspecified sentence.

One candidate objective for a query `q`, positive `p` and validated negatives
`N(q)` is:

```text
L_relation = -log(exp(score(q,p)/temperature)
                 / (exp(score(q,p)/temperature)
                    + sum_n exp(score(q,n)/temperature)))
```

The temperature is positive and dimensionless. Exclude alternative valid
positives from the negative set. Keep relation-specific labels and provenance;
do not repel antonyms in the association channel merely because they are
negatives in the equivalence channel. Test label errors and teacher disagreement.

[Counter-fitting](https://aclanthology.org/N16-1018/) supplies an established
precedent for using synonym and antonym constraints. The proposal extends this
principle to contextual senses and relation-specific channels. A pretrained
teacher contributes external knowledge and must be identified as such; gains
cannot be attributed solely to the user's corpus.

For sentence retrieval, retain an event structure such as
`predicate(agent, patient, polarity, qualifiers)`. Keep each event's bindings.
Pooling all agents and patients into separate global bags is still insufficient:
it loses which agent belongs to which event. Canonical roles can align active
and passive paraphrases while preserving participant reversal. Negation scope,
modality and quantifiers need their own represented fields or explicit limits.

### D. Compress only after the semantic representation passes

First compare the dense candidate representation against raw PPMI and the current
models. Then compare weighted sparse features, fixed-cardinality binary codes
and a dense index under a fixed memory/latency budget. Learn sparse-code rankings
from the relation objective or a validated teacher, rather than assuming random
projection supplies the missing meaning.

[SPLADE](https://arxiv.org/abs/2107.05720) is evidence that learned sparse
representations and inverted retrieval can be trained together. Its weighted
lexical features are not the same as Proteus's fixed-cardinality binary SDRs,
and its passage-retrieval results do not establish word-synonym quality.

Keep source vectors or richer evidence for reranking. The sparse candidate stage
must preserve recall; a reranker cannot recover candidates absent from its input.
Measure candidate recall and final precision independently. Learned bit indices
are not automatically human-readable concepts: attach representative evidence,
counterexamples and support counts before claiming interpretability.

## Candidate advance: evidence signatures with explicit contradiction

The most distinctive system-level proposal is to represent **what a passage
supports, what it explicitly contradicts, and what remains unknown**, conditioned
on sense, argument roles and scope. A single nonnegative overlap score cannot
express all three states. This is a proposed integration, not a claim of
unprecedented logic or a measured breakthrough.

For a declared, finite facet vocabulary, store positive and negative signatures
`D+` and `D-`, alongside the source assertions that justify each entry. Query
requirements similarly form `Q+` and `Q-`. Preserve event/argument bindings in
structured records; indexing role facets alone would still merge different
events. Given positive facet weights `w`:

```text
supported = weight((Q+ intersect D+) union (Q- intersect D-))
conflicted = weight((Q+ intersect D-) union (Q- intersect D+))
required = weight(Q+ union Q-)
unknown = required - supported - conflicted
score(query -> document) = (supported - lambda * conflicted) / required
```

This requires a nonempty query and consistent signatures: no facet may be in
both signs within the same scope. Contradictory sources must remain separately
attributed instead of being merged into a false consensus. `lambda >= 0` is a
declared penalty; the score is directional, dimensionless and lies in
`[-lambda, 1]`. Return support, conflict and unknown coverage as separate values,
not merely the combined number.

The potential gain is specific: shared subject matter no longer compensates
silently for an explicit contradiction. `hot` and `cold` may share a temperature
facet while disagreeing on its state. However, `not hot` must not be equated with
`cold`; neither absence nor an intermediate temperature establishes that
conclusion. A query about who blocks whom must compare bound arguments before
claiming support.

For lexical substitution, use contextual, bidirectional agreement as an
additional feature of the equivalence model, not as a universal synonym test.
For document retrieval, use sparse signatures to generate candidates and verify
their bound assertions during reranking. Teacher-generated assertions are
hypotheses about what the source states, not automatically established facts.

Falsification is direct: hold topical overlap fixed while changing a participant,
polarity or quantifier; require contradiction-sensitive ranking. Then test
meaning-preserving paraphrases to rule out a lexical mismatch detector. Measure
assertion extraction precision, unsupported assertions and retrieval coverage
separately. This could materially change semantic usefulness if the extraction
and calibration stages succeed; their accuracy is currently unknown.

## Two further opportunities

**Evidence-dependent confidence.** Estimate stability across independently
sampled documents or source groups, not tokens. For nonnegative source weights
`a_i` with a positive sum,
`n_eff = (sum_i a_i)^2 / sum_i a_i^2` is an effective weighted sample count under
an independence assumption. Duplicating a document should not increase
independent support. Distinguish ambiguity across senses from uncertainty caused
by insufficient evidence. Bootstrap stability alone is not calibrated accuracy.

**Separate a general prior from domain evidence.** A narrow corpus can inform
local usage without relearning all lexical knowledge. With aligned categorical
context features, a candidate shrinkage model is:

```text
p(context | word,sense) = (counts_domain + tau * p_prior)
                         / (domain_count + tau)
```

Here `tau >= 0` is prior strength in pseudo-observations. The domain counts sum
to `domain_count`, and the prior probabilities sum to one. A zero denominator
means the distribution is unknown. Unaligned embedding coordinates cannot be
averaged by this formula. Test both rare senses and strong domain shifts: a prior can erase
the very local meaning the corpus was intended to capture. Preserve the domain
evidence and prior as distinct provenance channels.

These are medium-priority proposals until the main semantic objective works.
Neither confidence estimation nor shrinkage fixes missing argument structure.

## Implementation order and acceptance experiments

| Order | Deliverable | Acceptance and falsification |
| --- | --- | --- |
| 1 | Occurrence/phrase records with typed contexts, plus a dense diagnostic path | Original offsets survive; reversal and active/passive cases are distinguishable as specified; compare directed-window and dependency baselines separately. |
| 2 | Sense-conditioned substitution and association models | Improve contextual substitution and sense discrimination; report failures by word frequency, sense and domain. Reject a model that improves only topical association. |
| 3 | Contrast supervision and general/domain separation | Measure antonym false positives and meaning-preserving substitutions on data excluded from teacher selection and tuning. |
| 4 | Sparse code training and retrieval cascade | Preserve dense-neighbor recall at the candidate stage and semantic metrics at the final stage under declared memory/latency limits. |

For future promotion, pre-register a practical improvement threshold such as an
absolute `+0.10` Spearman gain on a fresh similarity set, and a maximum `0.01`
loss from semantic vectors to the retrieval cascade. These are proposed
engineering thresholds, not measured outcomes or universal statistical cutoffs.
Use at least two corpus domains and multiple seeds; confidence intervals must
account for shared words, documents and repeated tuning.

[WiC](https://aclanthology.org/N19-1128/) tests whether the same word has the same
meaning in two contexts. [Swords](https://aclanthology.org/2021.naacl-main.345/)
tests contextual substitutions. Use their prescribed evaluation protocols and
verify data/teacher overlap. Include a separate expert-labeled application set:
benchmark substitution, lexical similarity and useful document retrieval are
related but different objectives.

Split synthetic and annotated sentence tests by template and lexical family,
not merely by row. Include participant swaps, active/passive paraphrases,
negation scope, multiword units and mixed-sense queries. Record all OOV items and
failed parses; report coverage alongside quality. The SimVerb and SimLex data
already inspected here are diagnostic/development evidence for future work.

Prospective model metadata needs a representation version, relation namespaces,
sense identities, corpus/extraction/teacher identities, and the scoring contract.
Old word-only retinas remain word-only; migration cannot reconstruct discarded
occurrences or sense provenance. Exact serialization design is deferred until an
algorithm earns promotion, not silently introduced by this theoretical review.

## Cost bounds

Let `N` be retained occurrences, `V` word types, `M` the maximum senses per word,
`d` dense coordinates, `I` clustering iterations, `P` training pairs, `G` sparse
feature slots, `K` active slots and `C` retrieved candidates.

| Stage | Work bound | Storage bound or qualification |
| --- | --- | --- |
| Context encoder | Depends on selected architecture and sequence lengths; measure separately | Model weights, activations and batching are additional costs. |
| Bounded per-word centroid clustering | O(I N M d) | O(N d + V M d) if all occurrence vectors are retained; streaming changes the algorithm. |
| Dense pair objective | O(P d) per pass for fixed sense assignments; O(P M^2 d) for all sense pairs | Excludes encoder backpropagation and negative mining. |
| Dense-to-sparse projection | O(V M G d) for a full dense projection | O(G d + V M K), excluding provenance and optional retained dense vectors. |
| Sense-aware reranking | O(C M^2 d) per word query | Candidate generation and context encoding are additional. |
| Sparse posting storage | O(V M K) entries | Byte cost depends on posting format, IDs, weights and metadata. |

At `V = 39141`, `d = 128`, dense binary32 coordinates alone occupy 20040192 bytes
(19.112 MiB). The existing angular file occupies 54587157 bytes (52.058 MiB),
including its index, vocabulary and metadata. This is not an equivalent-index
memory comparison, but it disproves any assumption that the present full file
is necessarily smaller than the raw coordinate array. Measure actual indexes.

## Assumption register

| ID | Assumption and basis | Dependent result | Stress test / falsification probe | Status |
| --- | --- | --- | --- | --- |
| T1 | Desired relevance includes equivalence and contextual meaning, inferred from the query failure and current request. | Priority of relation/sense redesign. | Compare user-labeled association and substitution tasks; reverse priority if association is the only required task. | Retained scope assumption. |
| T2 | The diagnostic path represents the frozen production models. | Attribution of pre-encoding failure. | Match normalized corpus hash and re-encode deterministic sampled words at every seed. | Confirmed for hashes and 65 sampled entries per seed; not exhaustive numerical independence. |
| T3 | SimVerb supplies relevant evidence about verb similarity. | Local quality conclusion. | Report all coverage, inspect overlap with prior evaluation and repeat on another domain/task. | Confirmed for this benchmark; application transfer unknown. |
| T4 | Additional contextual and labeled evidence can improve the target distinction. | Proposed semantic learner. | Matched corpus/encoder ablations with held-out relation labels; reject if only association improves. | Hypothesis supported by predecessors, untested here. |
| T5 | A bounded sense inventory captures useful ambiguity. | Multi-sense storage and scoring. | Rare senses, continuous sense variation and many-sense hubness; compare against occurrence retrieval. | Untested here. |
| T6 | Document/source groups provide adequate independent resampling units. | Confidence and shrinkage proposals. | Duplicated or correlated sources; check empirical confidence calibration. | Conditional assumption, untested here. |
| T7 | A sparse candidate index retains required semantic neighbors. | Efficient retrieval cascade. | Compare candidate recall with exhaustive dense/sense search before reranking. | Untested here. |

## Bounded adjacent findings

| Impact | Finding | Consequence and bound |
| --- | --- | --- |
| High | Static word pooling loses roles, polarity binding and senses. | Redesign represented evidence; extra optimizer iterations are insufficient. |
| High | Teacher knowledge and benchmark overlap can inflate measured progress. | Record training sources, freeze tests and split by word/source where appropriate. |
| High | Contextual parsing and sense induction can introduce new errors. | Retain raw text and alternatives; evaluate parse/sense failures explicitly. |
| Medium | Universal whitening or frequency cutoffs can remove useful evidence. | Validate spectral regularization and rare-word behavior separately. |
| Medium | More senses increase false-match opportunities and storage. | Bound senses, test posterior mixtures and measure candidate hubness. |
| Medium | Binary scores have nonzero chance overlap and correlated bits. | Calibrate against measured nulls and held-out labels, not the uniform-set formula alone. |
| Medium | A domain prior can overwhelm specialized meaning. | Evaluate prior-only, domain-only and combined models, retaining attribution. |
| Low | Display topology can be retained independently of retrieval. | Treat it as a visualization layer with its own fidelity tests. |

## Review gates

QG1: technical analysis only. QG2: material assumptions and falsification probes
are recorded above. QG3: theory, observed failure and actionable redesign are
covered. QG4: equations, units, costs and executable probes are explicit.
QG5: fixed-cardinality, coverage, corpus/feature confounding and benchmark reuse
are distinguished. QG6: primary references and reproducible local artifacts
support the claims. QG7: adjacent opportunities are bounded and labeled.

These gates apply to completion of the **review**. They do not certify the
unimplemented design, biological theory, future performance or scientific
novelty.
