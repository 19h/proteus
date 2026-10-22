# Scaling and cross-lingual research

[Documentation index](../README.md#documentation) · [Architecture](architecture.md)

This page condenses the original scaling and multilingual roadmap. The entries
below are research directions, not implemented services or delivery commitments.
The current CLI trains one in-memory corpus model; it has no distributed trainer,
multilingual model container, language router or REST server.

## Scaling beyond memory

Dense context vectors and SOM prototypes dominate current training cost. Mapped
retina queries do not make training stream from disk. See the
[cost model](algorithm.md#cost-and-bounded-extensions) before choosing grid and
feature dimensions; the [validation record](validation.md#corpus-execution)
contains one corpus experiment, not a capacity guarantee.

| Direction | Concrete work | Validation boundary |
| --- | --- | --- |
| Stream contexts | Sharded normalized snippets, reproducible vocabulary/document-frequency passes, bounded shuffle buffers | Match input boundaries and feature weights against the in-memory path |
| External feature storage | On-disk vectors, shard manifests and restart metadata | Measure resident memory, I/O and restart reproducibility |
| Parallelize exact work | Partition BMU distance work while preserving update order | Preserve tie-breaking and compare assignments and saved output |
| Distributed or batch updates | Define synchronization, update order and failure recovery | A changed optimizer needs a separate numerical and quality contract |
| Incremental learning | Vocabulary growth, model revisions and index rebuilds | Quantify drift and query compatibility across revisions |

The earlier throughput and corpus-size targets were proposals, not measured
results. No fixed terabyte-training capacity or linear scaling is established.

## Cross-lingual alignment

Cell indices from independently trained retinas have no shared meaning, even
when their grids have equal dimensions. Two candidate approaches remain:

| Approach | Required signal | Open issue |
| --- | --- | --- |
| Post-hoc map alignment | Bilingual anchors or aligned contexts | A rigid rotation/reflection need not align two learned, toroidal maps; measure residual distortion and neighbourhood preservation |
| Joint multilingual training | Shared feature space or explicitly coupled parallel contexts | Mixing unrelated token vocabularies alone supplies no translation alignment |

A multilingual model would also need per-language normalization and vocabulary
namespaces, language-aware query routing, and versioned alignment metadata.
Parallel text, translation dictionaries and linked concepts are possible inputs;
coverage, licensing and domain balance depend on the selected dataset.

Evaluation would separate held-out translation retrieval, topical retrieval and
analogy tasks, report the exact candidate vocabulary, and stratify by frequency,
language and sense. There is no current cross-lingual accuracy result.

## Assumption register and bounded opportunities

| ID / impact | Assumption or opportunity | Stress test |
| --- | --- | --- |
| R1 / high | Aligned contexts preserve enough shared semantic structure for cross-lingual retrieval | Hold out anchors, compare retrieval against unaligned and lexical baselines; alignment conclusions depend on this result |
| R2 / high | A changed training/storage regime preserves useful neighbourhoods | Compare quality, memory and runtime against the current exact path; throughput alone is insufficient |
| R3 / medium | One word fingerprint adequately covers its senses | Separate polysemous terms by context and compare single- versus multi-sense retrieval |
| R4 / medium | Tokenization gives comparable coverage across scripts and morphology | Measure unknown-token rate and retrieval by language, frequency and inflection |
| R5 / medium | Fingerprint topology can support model comparison or visualization | Align models first and test held-out neighbourhood stability; equal cell numbers are not an alignment |

Potential downstream work includes domain-specific retrieval, model-drift views
and sense-conditioned fingerprints. Each needs its own corpus and evaluation;
the presence of an SDR or a two-dimensional map does not establish its accuracy.
