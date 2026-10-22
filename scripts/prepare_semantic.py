#!/usr/bin/env python3
"""Prepare explicit lexical and parser evidence for Proteus semantic training.

Preparation uses optional Python packages. Native training, saved-model queries
and evaluation do not import Python or download models. Outputs record resource
identities and use UTF-8 byte offsets, not Python character offsets.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def output_stream(destination):
    """Publish a file only after successful preparation; preserve old contents."""
    if destination == "-":
        yield sys.stdout
        return
    path = Path(destination)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def prepare_lexicon(arguments):
    import nltk
    from nltk.corpus import wordnet

    data_directory = arguments.nltk_data.resolve()
    nltk.data.path.insert(0, str(data_directory))
    if arguments.download:
        nltk.download(
            "wordnet",
            download_dir=str(data_directory),
            quiet=True,
            raise_on_error=True,
        )
    archive = data_directory / "corpora" / "wordnet.zip"
    if not archive.is_file():
        raise ValueError("WordNet archive missing; supply --download or --nltk-data")
    wordnet.ensure_loaded()
    senses = []
    for sense in sorted(wordnet.all_synsets(), key=lambda item: item.name()):
        similar = {item.name() for item in sense.similar_tos() + sense.verb_groups()}
        similar.discard(sense.name())
        senses.append(
            {
                "id": sense.name(),
                "pos": sense.pos(),
                "definition": " ".join([sense.definition(), *sense.examples()]),
                "lemmas": [
                    {"text": lemma.name().replace("_", " "), "count": lemma.count()}
                    for lemma in sorted(sense.lemmas(), key=lambda item: item.name())
                ],
                "parents": sorted(
                    {
                        item.name()
                        for item in sense.hypernyms() + sense.instance_hypernyms()
                    }
                ),
                "similar": sorted(similar),
                "antonyms": sorted(
                    {
                        opposite.synset().name()
                        for lemma in sense.lemmas()
                        for opposite in lemma.antonyms()
                    }
                ),
            }
        )
    result = {
        "schema": 1,
        "source": {
            "name": "Princeton WordNet",
            "version": wordnet.get_version(),
            "url": "https://wordnet.princeton.edu/",
            "sha256": file_hash(archive),
            "license": wordnet.open("LICENSE").read(),
        },
        "senses": senses,
    }
    with output_stream(arguments.output) as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
    print(f"Prepared {len(senses)} lexical senses", file=sys.stderr)


def directory_hash(directory):
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_hash(path)))
    return digest.hexdigest()


def paragraphs(text):
    """Yield original paragraph text and its byte offset without reflowing it."""
    lines = []
    offset = 0
    start = 0
    for line in text.splitlines(keepends=True):
        if line.strip():
            if not lines:
                start = offset
            lines.append(line)
        elif lines:
            yield "".join(lines), start
            lines = []
        offset += len(line.encode("utf-8"))
    if lines:
        yield "".join(lines), start


def prepare_annotations(arguments):
    import spacy
    from thinc.api import get_current_ops

    raw = (
        arguments.input.read_bytes()
        if arguments.input is not None
        else arguments.text.encode("utf-8")
    )
    text = raw.decode("utf-8", errors="strict")
    source_hash = hashlib.sha256(raw).hexdigest()
    if arguments.device == "gpu":
        spacy.require_gpu()
    else:
        spacy.require_cpu()
    pipeline = spacy.load(arguments.model, disable=["ner"])
    required = {"parser", "lemmatizer"}
    if not required.issubset(pipeline.pipe_names):
        raise ValueError(
            "The selected spaCy pipeline must include parser and lemmatizer"
        )
    provenance = {
        "name": "spaCy",
        "version": importlib.metadata.version("spacy"),
        "model": arguments.model + ":" + pipeline.meta["version"],
        "model_sha256": directory_hash(Path(pipeline.path)),
        "source_sha256": source_hash,
        "backend": get_current_ops().name,
    }
    body_hash = hashlib.sha256()
    count = 0
    with output_stream(arguments.output) as stream:
        records = paragraphs(text)
        for parsed, (_, source_start) in pipeline.pipe(
            ((paragraph, (paragraph, offset)) for paragraph, offset in records),
            as_tuples=True,
            batch_size=arguments.batch_size,
        ):
            # spaCy indices count Unicode characters. Build one explicit map to
            # UTF-8 bytes so Rust slices and source references use the same units.
            byte_offsets = [0]
            for character in parsed.text:
                byte_offsets.append(byte_offsets[-1] + len(character.encode("utf-8")))
            sentence_ids = {}
            for sentence_id, sentence in enumerate(parsed.sents):
                for token in sentence:
                    sentence_ids[token.i] = sentence_id
            record = {
                "id": f"{source_hash}:{source_start}",
                "source_start": source_start,
                "text": parsed.text,
                "parser": provenance,
                "tokens": [
                    {
                        "start": byte_offsets[token.idx],
                        "end": byte_offsets[token.idx + len(token.text)],
                        "lemma": token.lemma_,
                        "pos": token.pos_,
                        "dependency": token.dep_,
                        "head": token.head.i,
                        "sentence": sentence_ids[token.i],
                        "morphology": (
                            str(token.morph).split("|") if token.morph else []
                        ),
                    }
                    for token in parsed
                ],
            }
            serialized = (
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            stream.write(serialized)
            body_hash.update(serialized.encode("utf-8"))
            count += 1
            if count % 1000 == 0:
                print(f"Parsed {count} documents", file=sys.stderr)
        if count == 0:
            raise ValueError("No nonempty documents")
        # A completion record also protects a native consumer from accepting a
        # successful prefix when a piped preparation process fails partway.
        stream.write(
            json.dumps(
                {
                    "schema": 1,
                    "complete": count,
                    "documents_sha256": body_hash.hexdigest(),
                }
            )
            + "\n"
        )
    print(f"Prepared {count} attributed documents", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    lexicon = commands.add_parser("lexicon", help="Export WordNet senses and relations")
    lexicon.add_argument("--nltk-data", type=Path, required=True)
    lexicon.add_argument("--download", action="store_true")
    lexicon.add_argument("--output", required=True)
    lexicon.set_defaults(handler=prepare_lexicon)

    annotations = commands.add_parser(
        "annotate", help="Preserve parsed corpus occurrences"
    )
    source = annotations.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--text")
    annotations.add_argument("--model", default="en_core_web_trf")
    annotations.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    annotations.add_argument("--batch-size", type=int, default=64)
    annotations.add_argument("--output", required=True)
    annotations.set_defaults(handler=prepare_annotations)
    arguments = parser.parse_args()
    if hasattr(arguments, "batch_size") and arguments.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        arguments.handler(arguments)
    except (OSError, ValueError, LookupError, ImportError) as error:
        parser.exit(2, f"Preparation failed: {error}\n")


if __name__ == "__main__":
    main()
