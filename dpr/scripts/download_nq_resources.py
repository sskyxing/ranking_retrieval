"""Plan or download the original DPR NQ resources with resume support.

The default is deliberately a dry run. Large downloads require --execute.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import sys
import urllib.error
import urllib.request


BASE = "https://dl.fbaipublicfiles.com/dpr"


@dataclass(frozen=True)
class Resource:
    name: str
    url: str
    relative_path: str
    expected_bytes: int


SMALL = (
    Resource(
        "nq-test",
        f"{BASE}/data/retriever/nq-test.qa.csv",
        "data/retriever/qas/nq-test.csv",
        296_949,
    ),
    Resource(
        "nq-dev",
        f"{BASE}/data/retriever/nq-dev.qa.csv",
        "data/retriever/qas/nq-dev.csv",
        620_013,
    ),
)

CHECKPOINT = (
    Resource(
        "paper-nq-checkpoint",
        f"{BASE}/checkpoint/retriever/single/nq/hf_bert_base.cp",
        "checkpoint/retriever/single/nq/bert-base-encoder.cp",
        875_965_155,
    ),
)

WIKI = (
    Resource(
        "wikipedia-passages-gzip",
        f"{BASE}/wikipedia_split/psgs_w100.tsv.gz",
        "compressed-data/wikipedia_split/psgs_w100.tsv.gz",
        4_694_541_059,
    ),
)

TRAIN = (
    Resource(
        "nq-train",
        f"{BASE}/data/retriever/biencoder-nq-train.json.gz",
        "compressed-data/retriever/biencoder-nq-train.json.gz",
        2_314_892_908,
    ),
    Resource(
        "nq-retriever-dev",
        f"{BASE}/data/retriever/biencoder-nq-dev.json.gz",
        "compressed-data/retriever/biencoder-nq-dev.json.gz",
        256_239_282,
    ),
)

PRECOMPUTED = (
    Resource(
        "nq-test-precomputed-retrieval",
        f"{BASE}/data/retriever_results/single/nq-test.json.gz",
        "compressed-data/retriever_results/single/nq-test.json.gz",
        87_432_793,
    ),
)

_EMBEDDING_SIZES = (
    1324693633, 1324804738, 1325065657, 1325225044, 1325225044,
    1325225044, 1325225044, 1325225044, 1325225044, 1325225044,
    1325225044, 1325225044, 1325225044, 1325225044, 1325225044,
    1325225044, 1325225044, 1325225044, 1325225044, 1325225044,
    1325225044, 1325225044, 1325225044, 1325312389, 1325645350,
    1325645350, 1325645350, 1325645350, 1325645350, 1325645350,
    1325645350, 1325645350, 1325645350, 1325645350, 1325645350,
    1325645350, 1325645350, 1325645350, 1325645350, 1325645350,
    1325645350, 1325645350, 1325645350, 1325645350, 1325645350,
    1325645350, 1325645350, 1325645350, 1325645350, 1325645350,
)
EMBEDDINGS = tuple(
    Resource(
        f"wikipedia-embeddings-{index:02d}",
        f"{BASE}/data/wiki_encoded/single/nq/wiki_passages_{index}",
        f"data/retriever_results/nq/single/wikipedia_passages_{index}.pkl",
        size,
    )
    for index, size in enumerate(_EMBEDDING_SIZES)
)

GROUPS = {
    "small": SMALL,
    "checkpoint": CHECKPOINT,
    "wiki": WIKI,
    "embeddings": EMBEDDINGS,
    "train": TRAIN,
    "precomputed": PRECOMPUTED,
}


def expand_groups(groups: list[str]) -> list[str]:
    expanded: list[str] = []
    for group in groups:
        expanded.extend(
            ("small", "checkpoint", "wiki", "embeddings")
            if group == "paper-eval"
            else ("small", "train") if group == "paper-train" else (group,)
        )
    return expanded


def select_resources(groups: list[str]) -> tuple[Resource, ...]:
    return tuple(
        dict.fromkeys(item for group in expand_groups(groups) for item in GROUPS[group])
    )


def human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def download(resource: Resource, root: Path) -> str:
    destination = root / resource.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == resource.expected_bytes:
        return "already complete"

    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > resource.expected_bytes:
        raise RuntimeError(f"partial file is too large: {partial}")

    request = urllib.request.Request(resource.url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        if error.code == 416 and offset == resource.expected_bytes:
            partial.replace(destination)
            return "completed from partial"
        raise

    status = getattr(response, "status", None)
    if offset and status != 206:
        response.close()
        offset = 0
        partial.unlink(missing_ok=True)
        response = urllib.request.urlopen(resource.url, timeout=60)

    mode = "ab" if offset else "wb"
    with response, partial.open(mode) as stream:
        shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
    actual = partial.stat().st_size
    if actual != resource.expected_bytes:
        raise RuntimeError(
            f"size mismatch for {resource.name}: expected "
            f"{resource.expected_bytes}, got {actual}; resumable file kept at {partial}"
        )
    partial.replace(destination)
    return "downloaded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        choices=tuple(GROUPS) + ("paper-eval", "paper-train"),
        help="Repeat to combine groups. Default: small (dry-run only).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "downloads"
    )
    parser.add_argument("--execute", action="store_true", help="Perform downloads.")
    parser.add_argument(
        "--manifest", type=Path, help="Optional JSON manifest path (written only with --execute)."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups = args.group or ["small"]
    resources = select_resources(groups)
    total = sum(item.expected_bytes for item in resources)
    print(f"Output: {args.output_dir.resolve()}")
    print(f"Resources: {len(resources)}; download size: {human_bytes(total)}")
    for item in resources:
        destination = args.output_dir / item.relative_path
        print(f"- {item.name}: {human_bytes(item.expected_bytes)} -> {destination}")

    if not args.execute:
        print("Dry run only. Add --execute to download.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(
                {"total_bytes": total, "resources": [asdict(item) for item in resources]},
                ensure_ascii=False, indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    usage = shutil.disk_usage(args.output_dir)
    if usage.free < total:
        print("Insufficient free disk space for selected downloads.", file=sys.stderr)
        return 2
    for index, item in enumerate(resources, 1):
        print(f"[{index}/{len(resources)}] {item.name}: {download(item, args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
