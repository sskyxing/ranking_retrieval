"""Stream DPR retrieval JSON and recompute answer-containing top-k accuracy."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import IO, Iterator
import unicodedata


SOURCE_URL = "https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-test.json.gz"


def tokenize(text: str) -> list[str]:
    """Match DPR's DrQA regex: L/N/M runs, then other non-Z/C chars."""
    normalized = unicodedata.normalize("NFD", text)
    tokens: list[str] = []
    index = 0
    while index < len(normalized):
        category = unicodedata.category(normalized[index])
        if category[0] in "LNM":
            end = index + 1
            while end < len(normalized) and unicodedata.category(normalized[end])[0] in "LNM":
                end += 1
            tokens.append(normalized[index:end].lower())
            index = end
        else:
            if category[0] not in "ZC":
                tokens.append(normalized[index].lower())
            index += 1
    return tokens


def contains_answer(text: str, answers: list[str]) -> bool:
    text_tokens = tokenize(text)
    for answer in answers:
        answer_tokens = tokenize(answer)
        for start in range(len(text_tokens) - len(answer_tokens) + 1):
            if answer_tokens == text_tokens[start : start + len(answer_tokens)]:
                return True
    return False


def iter_json_array(stream: IO[str], chunk_size: int = 1024 * 1024) -> Iterator[dict]:
    """Decode a top-level JSON array without retaining the whole file."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    finished = False

    while not finished:
        if position >= len(buffer) - 1:
            buffer = buffer[position:] + stream.read(chunk_size)
            position = 0
            if not buffer:
                break
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if not started:
            if position >= len(buffer) or buffer[position] != "[":
                raise ValueError("expected a top-level JSON array")
            position += 1
            started = True
            continue
        while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
            position += 1
        if position < len(buffer) and buffer[position] == "]":
            finished = True
            position += 1
            continue
        try:
            item, end = decoder.raw_decode(buffer, position)
        except json.JSONDecodeError:
            remainder = buffer[position:]
            more = stream.read(chunk_size)
            if not more:
                raise ValueError("truncated JSON array")
            buffer = remainder + more
            position = 0
            continue
        if not isinstance(item, dict):
            raise ValueError("expected each array element to be an object")
        yield item
        position = end
        if position > chunk_size:
            buffer = buffer[position:]
            position = 0

    if not finished:
        raise ValueError("JSON array did not terminate")
    if stream.read(1).strip():
        raise ValueError("unexpected data after JSON array")


def evaluate(path: Path, cutoffs: tuple[int, ...], recompute: bool) -> dict:
    cached_hits = {cutoff: 0 for cutoff in cutoffs}
    recomputed_hits = {cutoff: 0 for cutoff in cutoffs}
    disagreement_contexts = 0
    questions = 0
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for sample in iter_json_array(stream):
            contexts = sample.get("ctxs")
            if not isinstance(contexts, list):
                raise ValueError(f"sample {questions} has no ctxs list")
            cached_flags = [bool(context.get("has_answer")) for context in contexts]
            if recompute:
                answers = sample.get("answers")
                if not isinstance(answers, list) or not all(isinstance(item, str) for item in answers):
                    raise ValueError(f"sample {questions} has no valid answers list")
                recomputed_flags = [
                    contains_answer(str(context.get("text", "")), answers)
                    for context in contexts
                ]
                disagreement_contexts += sum(
                    left != right for left, right in zip(cached_flags, recomputed_flags)
                )
            else:
                recomputed_flags = cached_flags
            questions += 1
            for cutoff in cutoffs:
                cached_hits[cutoff] += int(any(cached_flags[:cutoff]))
                recomputed_hits[cutoff] += int(any(recomputed_flags[:cutoff]))
    if not questions:
        raise ValueError("no questions found")
    def render(hits: dict[int, int]) -> dict:
        return {
            str(cutoff): {
                "hits": hits[cutoff],
                "fraction": hits[cutoff] / questions,
                "percent": 100.0 * hits[cutoff] / questions,
            }
            for cutoff in cutoffs
        }

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    report = {
        "source": str(path.resolve()),
        "source_url": SOURCE_URL,
        "source_bytes": path.stat().st_size,
        "source_sha256": digest.hexdigest(),
        "question_count": questions,
        "cached_has_answer_top_k_accuracy": render(cached_hits),
    }
    if recompute:
        report["recomputed_top_k_accuracy"] = render(recomputed_hits)
        report["cached_vs_recomputed_context_disagreements"] = disagreement_contexts
        report["tokenizer_note"] = (
            "Standard-library implementation of the paper-era DrQA SimpleTokenizer: "
            "NFD, Unicode L/N/M runs, individual other non-Z/C characters, lowercase"
        )
    return report


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?", type=Path,
        default=base / "downloads/compressed-data/retriever_results/single/nq-test.json.gz",
    )
    parser.add_argument("--cutoffs", type=int, nargs="+", default=(1, 5, 20, 100))
    parser.add_argument(
        "--trust-cached-flags", action="store_true",
        help="Skip recomputing answer containment from answers and passage text.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cutoffs = tuple(sorted(set(args.cutoffs)))
    if any(cutoff < 1 for cutoff in cutoffs):
        raise ValueError("cutoffs must be positive")
    report = evaluate(args.path, cutoffs, recompute=not args.trust_cached_flags)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
