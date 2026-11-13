"""Stream-extract DPR NQ archives after download; default is a dry run."""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path
import shutil


ARCHIVES = {
    "wiki": ("compressed-data/wikipedia_split/psgs_w100.tsv.gz", "data/wikipedia_split/psgs_w100.tsv"),
    "train": ("compressed-data/retriever/biencoder-nq-train.json.gz", "data/retriever/nq-train.json"),
    "dev": ("compressed-data/retriever/biencoder-nq-dev.json.gz", "data/retriever/nq-dev.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource", action="append", choices=tuple(ARCHIVES) + ("all",), required=True)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1] / "downloads"
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names = list(ARCHIVES) if "all" in args.resource else list(dict.fromkeys(args.resource))
    pairs = [(name, args.root / ARCHIVES[name][0], args.root / ARCHIVES[name][1]) for name in names]
    for name, source, target in pairs:
        print(f"- {name}: {source} -> {target}")
        if not source.is_file():
            raise FileNotFoundError(f"missing archive: {source}")
    if not args.execute:
        print("Dry run only. Add --execute to extract.")
        return 0

    for name, source, target in pairs:
        if target.exists():
            print(f"{name}: already exists; skipped")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        partial.unlink(missing_ok=True)
        try:
            with gzip.open(source, "rb") as compressed, partial.open("wb") as output:
                shutil.copyfileobj(compressed, output, length=8 * 1024 * 1024)
            partial.replace(target)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        print(f"{name}: extracted {target.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
