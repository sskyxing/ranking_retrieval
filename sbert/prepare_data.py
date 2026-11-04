"""Prepare historical AllNLI, official reference TSVs, and a pinned BERT checkpoint."""

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import gzip
import hashlib
import io
from itertools import zip_longest
import json
from pathlib import Path
import urllib.request
import zipfile

BASE = Path(__file__).resolve().parent
DATASETS = {
    "AllNLI.tsv.gz": "https://sbert.net/datasets/AllNLI.tsv.gz",
    "stsbenchmark.tsv.gz": "https://sbert.net/datasets/stsbenchmark.tsv.gz",
}
MODEL_ID = "google-bert/bert-base-uncased"
MODEL_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
HISTORICAL_ARCHIVE = "AllNLI-original.zip"
HISTORICAL_TSV = "AllNLI-historical.tsv.gz"
HISTORICAL_URL = ("https://public.ukp.informatik.tu-darmstadt.de/"
                  "reimers/sentence-transformers/datasets/AllNLI.zip")
HISTORICAL_SHA256 = "fe2d8194a49869667196f7c991b9567dd835c75f35a972d6b7f3e06557e683ff"
HISTORICAL_SPLITS = {"train": 942069, "dev": 29489}
EXPECTED = {
    "AllNLI": ({"train": 942069, "dev": 19657, "test": 19656},
               "60abeaa8f94820954297520a759e677cce6931f524cf535408dee0aab5ae7b54"),
    "stsbenchmark": ({"train": 5749, "dev": 1500, "test": 1379},
                     "aa3ae9b0060fdab8a7239df4bed7668086b4181e7cf6e921d086a70f9e42c17e"),
}
MODEL_FILES = ["config.json", "model.safetensors", "tokenizer.json",
               "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_tsv(path, dataset=None):
    counts = Counter()
    labels = Counter()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t", quoting=csv.QUOTE_NONE)
        required = {"split", "sentence1", "sentence2"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Invalid dataset columns in {path}")
        for row in reader:
            if not row["sentence1"] or not row["sentence2"]:
                raise ValueError(f"Empty sentence in {path}")
            counts[row["split"]] += 1
            if "label" in row:
                if row["label"] not in {"contradiction", "entailment", "neutral"}:
                    raise ValueError(f"Unknown NLI label: {row['label']}")
                labels[row["label"]] += 1
            else:
                score = float(row["score"])
                if not 0 <= score <= 5:
                    raise ValueError(f"Invalid STS score: {score}")
    if dataset is None:
        dataset = ("AllNLI-historical" if path.name.startswith("AllNLI-historical")
                   else "AllNLI" if path.name.startswith("AllNLI") else "stsbenchmark")
    expected_splits = HISTORICAL_SPLITS if dataset == "AllNLI-historical" else EXPECTED[dataset][0]
    if dict(counts) != expected_splits:
        raise ValueError(f"Unexpected split counts in {path}: {dict(counts)}")
    return {"splits": dict(counts), "labels": dict(labels)}


def historical_field(line):
    """Remove one line terminator without changing original sentence whitespace."""
    value = line.removesuffix("\n").removesuffix("\r")
    if any(character in value for character in ("\t", "\r", "\n")):
        raise ValueError("Historical AllNLI contains a tab or embedded line terminator")
    return value


def prepare_historical(data_dir, previous):
    archive_path = data_dir / HISTORICAL_ARCHIVE
    if not archive_path.exists():
        partial = archive_path.with_suffix(archive_path.suffix + ".partial")
        print(f"Downloading {HISTORICAL_URL}", flush=True)
        request = urllib.request.Request(HISTORICAL_URL, headers={"User-Agent": "SBERT-reproduction/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        if sha256(partial) != HISTORICAL_SHA256:
            raise ValueError("Historical AllNLI archive SHA256 differs from pinned copy")
        partial.replace(archive_path)
    if sha256(archive_path) != HISTORICAL_SHA256:
        raise ValueError(f"Historical AllNLI archive SHA256 differs from pinned copy: {archive_path}")

    destination = data_dir / HISTORICAL_TSV
    partial = destination.with_suffix(destination.suffix + ".partial")
    members = [f"AllNLI/{column}.{split}.gz"
               for split in HISTORICAL_SPLITS for column in ("labels", "s1", "s2")]
    try:
        with zipfile.ZipFile(archive_path) as archive, partial.open("wb") as raw:
            # An empty filename and zero mtime keep the gzip header deterministic.
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0) as compressed:
                compressed.write(b"split\tlabel\tsentence1\tsentence2\n")
                for split in HISTORICAL_SPLITS:
                    with ExitStack() as stack:
                        streams = []
                        for column in ("labels", "s1", "s2"):
                            member = f"AllNLI/{column}.{split}.gz"
                            zipped = stack.enter_context(archive.open(member, "r"))
                            uncompressed = stack.enter_context(gzip.GzipFile(fileobj=zipped, mode="rb"))
                            streams.append(stack.enter_context(io.TextIOWrapper(
                                uncompressed, encoding="utf-8", newline="",
                            )))
                        sentinel = object()
                        for row in zip_longest(*streams, fillvalue=sentinel):
                            if any(value is sentinel for value in row):
                                raise ValueError(f"Historical AllNLI columns have different lengths: {split}")
                            label, sentence1, sentence2 = (historical_field(value) for value in row)
                            if label not in {"contradiction", "entailment", "neutral"}:
                                raise ValueError(f"Unknown historical NLI label: {label!r}")
                            if not sentence1 or not sentence2:
                                raise ValueError(f"Empty historical AllNLI sentence in {split}")
                            compressed.write(("\t".join((split, label, sentence1, sentence2)) + "\n").encode("utf-8"))
        details = inspect_tsv(partial, dataset="AllNLI-historical")
        checksum = sha256(partial)
        old = previous.get("datasets", {}).get(HISTORICAL_TSV, {}).get("sha256")
        if old is not None and checksum != old:
            raise ValueError("Historical AllNLI deterministic export SHA256 changed")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    result = {
        HISTORICAL_ARCHIVE: {
            "url": HISTORICAL_URL, "sha256": HISTORICAL_SHA256,
            "members": members, "splits": details["splits"],
        },
        HISTORICAL_TSV: {
            "source_archive": HISTORICAL_ARCHIVE, "source_sha256": HISTORICAL_SHA256,
            "sha256": checksum, "gzip_mtime": 0,
            "preprocessing": "Original sentences; remove only line terminators; no detokenization",
            **details,
        },
    }
    for filename, detail in result.items():
        print(f"{filename}: {detail}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE / "data")
    parser.add_argument("--model-dir", type=Path, default=BASE / "models" / "bert-base-uncased")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.data_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest = {"datasets": {}, "model": {"id": MODEL_ID, "revision": MODEL_REVISION}}
    for filename, url in DATASETS.items():
        path = args.data_dir / filename
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".partial")
            print(f"Downloading {url}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "SBERT-reproduction/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
            inspect_tsv(partial)
            partial.replace(path)
        checksum = sha256(path)
        dataset = "AllNLI" if filename.startswith("AllNLI") else "stsbenchmark"
        if checksum != EXPECTED[dataset][1]:
            raise ValueError(f"Official dataset SHA256 differs from pinned copy: {path}")
        old = previous.get("datasets", {}).get(filename, {}).get("sha256")
        if old is not None and checksum != old:
            raise ValueError(f"Dataset checksum changed: {path}")
        manifest["datasets"][filename] = {"url": url, "sha256": checksum, **inspect_tsv(path)}
        print(f"{filename}: {manifest['datasets'][filename]}", flush=True)
    manifest["datasets"].update(prepare_historical(args.data_dir, previous))
    atomic_json(manifest_path, manifest)
    if not args.data_only:
        from huggingface_hub import snapshot_download

        args.model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {MODEL_ID}@{MODEL_REVISION}", flush=True)
        snapshot_download(
            repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=str(args.model_dir),
            allow_patterns=MODEL_FILES,
        )
        files = {path.name: sha256(path) for path in args.model_dir.iterdir()
                 if path.is_file() and path.name in MODEL_FILES}
        if not {"config.json", "model.safetensors", "tokenizer.json"}.issubset(files):
            raise ValueError("BERT download incomplete")
        provenance = {"id": MODEL_ID, "revision": MODEL_REVISION, "sha256": files}
        atomic_json(args.model_dir / "source.json", provenance)
        print(f"Model ready: {args.model_dir}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
