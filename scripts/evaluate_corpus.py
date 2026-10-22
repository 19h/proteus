#!/usr/bin/env python3
"""Reproducible CLI smoke evaluation; no unlabelled semantic-quality assertion.

Example:
  python3 scripts/evaluate_corpus.py corpus.txt target/validation --dimension 32
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("corpus", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--binary", type=Path, default=Path("target/release/proteus"))
p.add_argument("--dimension", type=int, default=32)
p.add_argument("--iterations", type=int, default=5000)
p.add_argument("--projection-dim", type=int, default=256)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--words", nargs="+", default=["king", "queen", "whale", "ship"])
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
binary_sha256 = hashlib.sha256(args.binary.read_bytes()).hexdigest()
model = args.output / f"d{args.dimension}-p{args.projection_dim}-s{args.seed}.retina"
command = [str(args.binary.resolve()), "train", "-i", str(args.corpus), "-o", str(model),
           "-d", str(args.dimension), "-n", str(args.iterations), "--seed", str(args.seed),
           "--projection-dim", str(args.projection_dim)]
start = time.perf_counter()
train = subprocess.run(command, capture_output=True, text=True, check=True)
elapsed = time.perf_counter() - start
queries = {}
for word in args.words:
    query = [str(args.binary.resolve()), "similar", "-r", str(model), word]
    outputs = [subprocess.run(query, capture_output=True, text=True, check=True).stdout for _ in range(10)]
    assert len(set(outputs)) == 1, f"Unstable reload/query output for {word}"
    queries[word] = outputs[0]
assert binary_sha256 == hashlib.sha256(args.binary.read_bytes()).hexdigest(), "Binary changed during evaluation"
report = {
    "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
    "binary_sha256": binary_sha256,
    "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
    "command": command, "elapsed_seconds": elapsed, "training_output": train.stdout,
    "model_info": subprocess.run([str(args.binary.resolve()), "info", str(model)],
                                 capture_output=True, text=True, check=True).stdout,
    "reloads_per_word": 10, "queries": queries,
    "scope": "CLI execution and repeated-load determinism; unlabelled corpus, no accuracy estimate",
}
report_path = model.with_suffix(".json")
report_path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
