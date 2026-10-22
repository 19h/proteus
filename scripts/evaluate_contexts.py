#!/usr/bin/env python3
"""Evaluate WiC contexts natively, retaining coverage and lexical-source overlap.

The caller supplies local data and optional labels. No test split is opened
implicitly, and no dataset or neural resource is downloaded by this tool.
"""

import argparse
import hashlib
import json
import random
import re
import subprocess
import unicodedata
from pathlib import Path

from prepare_semantic import file_hash, output_stream


def target_span(text, position):
    words = list(re.finditer(r"\S+", text))
    if position < 0 or position >= len(words):
        raise ValueError(f"Target index {position} is outside its context")
    match = words[position]
    return {
        "start": len(text[: match.start()].encode("utf-8")),
        "end": len(text[: match.end()].encode("utf-8")),
    }


def read_wic(data_path, gold_path=None):
    lines = data_path.read_bytes().decode("utf-8", errors="strict").splitlines()
    labels = (
        gold_path.read_bytes().decode("utf-8", errors="strict").splitlines()
        if gold_path
        else [None] * len(lines)
    )
    if len(labels) != len(lines) or any(
        label not in {None, "T", "F"} for label in labels
    ):
        raise ValueError(
            "WiC labels must contain exactly one T/F observation per data row"
        )
    pairs = []
    observations = []
    for row_number, (line, label) in enumerate(zip(lines, labels), 1):
        fields = line.split("\t")
        if len(fields) != 5:
            raise ValueError(
                f"WiC row {row_number} must contain five tab-separated fields"
            )
        word, pos, positions, first, second = fields
        if not word or pos not in {"N", "V"}:
            raise ValueError(f"Invalid target or POS at WiC row {row_number}")
        indices = positions.split("-")
        if len(indices) != 2:
            raise ValueError(f"Invalid target positions at WiC row {row_number}")
        indices = [int(index) for index in indices]
        usages = [
            {
                "word": word,
                "context": text,
                "part_of_speech": {"N": "noun", "V": "verb"}[pos],
                "target": target_span(text, index),
                "target_lemma": word,
            }
            for text, index in zip([first, second], indices)
        ]
        pairs.append(
            {
                "id": str(row_number),
                "left": usages[0],
                "right": usages[1],
                "same_sense": None if label is None else label == "T",
            }
        )
        observations.append({"id": str(row_number), "word": word, "pos": pos})
    return pairs, observations


def normalized_words(text):
    text = unicodedata.normalize("NFD", text).lower()
    return " ".join(re.findall(r"[^\W_]+", text))


def lexical_overlap(pairs, path, expected_hash):
    lexicon = json.loads(path.read_bytes())
    # Match the public Rust structs' field order, independent of JSON key order.
    canonical = {
        "schema": lexicon["schema"],
        "source": {
            key: lexicon["source"][key]
            for key in ["name", "version", "url", "sha256", "license"]
        },
        "senses": [
            {
                key: (
                    sense[key]
                    if key != "lemmas"
                    else [
                        {"text": lemma["text"], "count": lemma["count"]}
                        for lemma in sense["lemmas"]
                    ]
                )
                for key in [
                    "id",
                    "pos",
                    "definition",
                    "lemmas",
                    "parents",
                    "similar",
                    "antonyms",
                ]
            }
            for sense in lexicon["senses"]
        ],
    }
    digest = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if digest != expected_hash:
        raise ValueError(
            "Overlap-audit lexical source differs from the retina's canonical source"
        )
    lemmas = {}
    for sense in lexicon["senses"]:
        gloss = " " + normalized_words(sense["definition"]) + " "
        for lemma in sense["lemmas"]:
            lemmas.setdefault(normalized_words(lemma["text"]), []).append(gloss)
    output = []
    for pair in pairs:
        matches = []
        for side in ["left", "right"]:
            query = pair[side]
            phrase = normalized_words(query["context"])
            matches.append(
                bool(phrase)
                and any(
                    " " + phrase + " " in gloss
                    for gloss in lemmas.get(normalized_words(query["word"]), [])
                )
            )
        output.append(matches)
    return output


def summary(rows):
    common = [
        row
        for row in rows
        if row["gold"] is not None
        and row["baseline"] is not None
        and row["predicted"] is not None
    ]
    return {
        "total_pairs": len(rows),
        "common_labelled_pairs": len(common),
        "baseline_correct": sum(row["baseline"] == row["gold"] for row in common),
        "candidate_correct": sum(row["predicted"] == row["gold"] for row in common),
    }


def paired_cluster_bootstrap(rows, observations, draws, seed):
    groups = {}
    for row, observation in zip(rows, observations):
        if (
            row["gold"] is not None
            and row["baseline"] is not None
            and row["predicted"] is not None
        ):
            groups.setdefault(observation["word"], []).append(
                int(row["predicted"] == row["gold"])
                - int(row["baseline"] == row["gold"])
            )
    if not groups:
        return None
    words = sorted(groups)
    generator = random.Random(seed)
    differences = []
    for _ in range(draws):
        sample = [groups[generator.choice(words)] for _ in words]
        differences.append(
            sum(sum(group) for group in sample) / sum(len(group) for group in sample)
        )
    differences.sort()
    return {
        "clusters": len(words),
        "draws": draws,
        "seed": seed,
        "delta_accuracy_95_percentile_interval": [
            differences[int((draws - 1) * q)] for q in [0.025, 0.975]
        ],
        "scope": "Descriptive paired bootstrap by target lemma; shared resources and relations can still induce dependence between clusters.",
    }


def run(arguments):
    sources = {
        "binary": arguments.binary,
        "retina": arguments.retina,
        "data": arguments.data,
    }
    if arguments.gold:
        sources["gold"] = arguments.gold
    if arguments.lexicon:
        sources["lexicon"] = arguments.lexicon
    pairs_path = arguments.output.with_suffix(".pairs.jsonl")
    destinations = {arguments.output.resolve(), pairs_path.resolve()}
    if any(path.resolve() in destinations for path in sources.values()):
        raise ValueError("Evaluation outputs must not replace an input source")
    identities = {
        name: {"path": str(path), "sha256": file_hash(path)}
        for name, path in sources.items()
    }
    pairs, observations = read_wic(arguments.data, arguments.gold)
    with output_stream(str(pairs_path)) as stream:
        for pair in pairs:
            stream.write(json.dumps(pair, ensure_ascii=False) + "\n")
    command = [
        str(arguments.binary.resolve()),
        "semantic",
        "compare-contexts",
        "-r",
        str(arguments.retina),
        "--pairs",
        str(pairs_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    native = json.loads(process.stdout)
    rows = native["evaluation"]["predictions"]
    if len(rows) != len(observations) or any(
        row["id"] != observation["id"] for row, observation in zip(rows, observations)
    ):
        raise ValueError("Native evaluation did not retain the complete ordered input")
    overlap = (
        lexical_overlap(
            pairs, arguments.lexicon, native["evaluation"]["lexicon_sha256"]
        )
        if arguments.lexicon
        else None
    )
    strata = {}
    if overlap is not None:
        for name, predicate in [
            ("neither_context_in_source", lambda flags: not any(flags)),
            ("one_or_both_contexts_in_source", any),
            ("both_contexts_in_source", all),
        ]:
            strata[name] = summary(
                [row for row, flags in zip(rows, overlap) if predicate(flags)]
            )
    for name, path in sources.items():
        if file_hash(path) != identities[name]["sha256"]:
            raise ValueError(f"Evaluation source changed during execution: {name}")
    result = {
        "inputs": identities,
        "command": command,
        "native": native,
        "observations": observations,
        "source_context_overlap": overlap,
        "source_overlap_strata": strata,
        "paired_cluster_bootstrap": paired_cluster_bootstrap(
            rows, observations, arguments.bootstrap_draws, arguments.seed
        ),
        "scope": "Same-MAP-sense diagnostic. Literal resource-context overlap is reported explicitly; held-out pair labels do not establish resource-disjoint contextual generalization.",
    }
    with output_stream(str(arguments.output)) as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in native["evaluation"].items()
                if key != "predictions"
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/proteus"))
    parser.add_argument("--retina", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--lexicon", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    if arguments.bootstrap_draws < 1:
        parser.error("--bootstrap-draws must be positive")
    run(arguments)


if __name__ == "__main__":
    main()
