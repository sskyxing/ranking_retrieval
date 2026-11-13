"""Evaluate a DPR checkpoint with the BM25 NFCorpus metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizer


def load_protocol(data_dir: Path, raw_text: bool):
    baseline_repo = data_dir.resolve().parent
    if not (baseline_repo / "bm25" / "run_bm25.py").is_file():
        raise FileNotFoundError(f"BM25 metric implementation is missing under {baseline_repo}")
    if not (baseline_repo / "sbert" / "retrieval_data.py").is_file():
        raise FileNotFoundError(f"SBERT raw-text loader is missing under {baseline_repo}")
    sys.path.insert(0, str(baseline_repo))
    from bm25.run_bm25 import evaluate
    from sbert.retrieval_data import load_split

    documents, queries, relevance = load_split(data_dir, "test", raw_text=raw_text)
    if len(documents) != 3162 or len(queries) != 325 or len(relevance) != 323:
        raise ValueError("Unexpected NFCorpus test split; confirm the supplied baseline data.")
    return documents, queries, relevance, evaluate


def load_encoders(checkpoint_path: Path, return_similarity_mode: bool = False):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint["state_dict"]
    similarity_mode = checkpoint.get("hyper_parameters", {}).get("similarity_mode", "dot")
    if similarity_mode not in ("dot", "cosine"):
        raise ValueError(f"Unsupported checkpoint similarity mode: {similarity_mode}")
    models = []
    for tower in ("question", "context"):
        prefix = f"model.{tower}_encoder."
        tower_state = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
        if not tower_state:
            raise ValueError(f"Checkpoint lacks the DPR {tower} tower.")
        model = BertModel(BertConfig())
        model.load_state_dict(tower_state, strict=True)
        models.append(model.eval())
    result = models[0], models[1], int(checkpoint["epoch"]), int(checkpoint["global_step"])
    return (*result, similarity_mode) if return_similarity_mode else result


def score_passages(question_vectors: torch.Tensor, passage_vectors: torch.Tensor, similarity_mode: str):
    """Score with the same vector preparation used during checkpoint training."""

    if similarity_mode == "cosine":
        question_vectors = F.normalize(question_vectors, p=2, dim=-1)
        passage_vectors = F.normalize(passage_vectors, p=2, dim=-1)
    elif similarity_mode != "dot":
        raise ValueError(f"Unsupported checkpoint similarity mode: {similarity_mode}")
    return question_vectors @ passage_vectors.T


def context_rows(documents: Dict[str, str], tokenizer: BertTokenizer, mode: str, max_length: int, overlap: int):
    doc_ids = sorted(documents)
    rows: List[List[int]] = []
    row_doc_indices: List[int] = []
    counts = []
    for doc_index, doc_id in enumerate(doc_ids):
        text = documents[doc_id]
        if mode == "preprocessed-single":
            chunks = [tokenizer.encode(text, truncation=True, max_length=max_length)]
        else:
            title, separator, abstract = text.partition("\n")
            if not separator:
                raise ValueError(f"Raw document {doc_id} has no title/abstract separator.")
            if mode == "raw-single":
                chunks = [tokenizer.encode(title, text_pair=abstract, truncation=True, max_length=max_length)]
            else:
                title_ids = tokenizer.encode(title, add_special_tokens=False, verbose=False)[: max_length - 4]
                abstract_ids = tokenizer.encode(abstract, add_special_tokens=False, verbose=False)
                capacity = max_length - len(title_ids) - 3
                stride = capacity - min(overlap, capacity - 1)
                chunks = []
                for start in range(0, max(1, len(abstract_ids)), stride):
                    chunks.append(
                        tokenizer.build_inputs_with_special_tokens(title_ids, abstract_ids[start : start + capacity])
                    )
                    if start + capacity >= len(abstract_ids):
                        break
        rows.extend(chunks)
        row_doc_indices.extend([doc_index] * len(chunks))
        counts.append(len(chunks))
    return (
        doc_ids,
        rows,
        row_doc_indices,
        {
            "passage_count": len(rows),
            "max_passages_per_document": max(counts),
            "documents_with_multiple_passages": sum(count > 1 for count in counts),
            "passage_count_histogram": dict(sorted(Counter(counts).items())),
        },
    )


@torch.inference_mode()
def encode(model: BertModel, tokenizer: BertTokenizer, rows: List[List[int]], batch_size: int):
    vectors = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        features = tokenizer.pad(
            [{"input_ids": row, "attention_mask": [1] * len(row)} for row in batch],
            padding=True,
            return_tensors="pt",
        )
        features["token_type_ids"] = torch.zeros_like(features["input_ids"])
        vectors.append(model(**features, return_dict=True).last_hidden_state[:, 0, :].cpu())
    return torch.cat(vectors, dim=0)


class PrecomputedRetriever:
    def __init__(self, queries: Dict[str, str], doc_ids: List[str], scores: torch.Tensor):
        if scores.shape != (len(queries), len(doc_ids)):
            raise ValueError("Query/document score matrix has the wrong shape.")
        self.ranked = {}
        for text, row in zip(queries.values(), scores):
            # Sorted IDs plus stable score sorting reproduce the BM25 tie rule.
            ranking = torch.argsort(row, descending=True, stable=True)[:10]
            self.ranked[text] = [doc_ids[index] for index in ranking.tolist()]

    def search(self, query: str, k: int = 10) -> List[str]:
        return self.ranked[query][:k]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_report_config(
    args, epoch: int, global_step: int, data_names: List[str], passage_stats: Dict, similarity_mode: str = "dot"
):
    """Describe the evaluated checkpoint and protocol without assuming its training history."""

    return {
        "checkpoint": str(args.checkpoint.resolve()),
        "source_aml_run_id": args.source_aml_run_id,
        "checkpoint_epoch": epoch,
        "checkpoint_global_step": global_step,
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "similarity_mode": similarity_mode,
        "training_data": args.training_data.strip(),
        "text_mode": args.text_mode,
        "text_source": (
            "raw query title; raw document title + abstract"
            if args.text_mode != "preprocessed-single"
            else "test.titles.queries; test.docs"
        ),
        "document_score": (
            "maximum L2-normalized CLS cosine similarity across passages"
            if similarity_mode == "cosine"
            else "maximum unnormalized CLS dot product across passages"
        ),
        "max_length": args.max_length,
        "overlap": args.overlap if args.text_mode == "raw-chunks" else None,
        "tokenizer": args.tokenizer,
        "query_preprocessing": "remove one trailing question mark, as in DPR training",
        "tie_break": "score descending, doc_id ascending",
        "metrics": "BM25 baseline evaluate(): macro Recall@10 and linear-gain nDCG@10",
        "data_sha256": {name: sha256(args.data_dir / name) for name in data_names},
        **passage_stats,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--training-data",
        required=True,
        help="Human-readable training provenance, for example 'NQ warm start; NFCorpus train fine-tuning'.",
    )
    parser.add_argument(
        "--source-aml-run-id",
        default=None,
        help="Optional AML run ID that produced the evaluated checkpoint.",
    )
    parser.add_argument(
        "--text-mode", choices=("raw-chunks", "raw-single", "preprocessed-single"), default="raw-chunks"
    )
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1 or not 8 <= args.max_length <= 512 or args.overlap < 0:
        parser.error("batch-size and threads must be positive; max-length must be 8..512; overlap must be nonnegative")
    if not args.training_data.strip():
        parser.error("training-data must be nonempty")
    if args.source_aml_run_id is not None and not args.source_aml_run_id.strip():
        parser.error("source-aml-run-id must be nonempty when supplied")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)

    torch.set_num_threads(args.threads)
    documents, queries, relevance, evaluate = load_protocol(args.data_dir, args.text_mode != "preprocessed-single")
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer, do_lower_case=True)
    question_model, context_model, epoch, global_step, similarity_mode = load_encoders(
        args.checkpoint, return_similarity_mode=True
    )
    query_ids = sorted(queries)
    query_index = {query_id: index for index, query_id in enumerate(query_ids)}
    # Match the DPR training data loader: remove one trailing question mark.
    question_rows = [
        tokenizer.encode(
            queries[qid][:-1] if queries[qid].endswith("?") else queries[qid],
            truncation=True,
            max_length=args.max_length,
        )
        for qid in query_ids
    ]
    question_vectors = encode(question_model, tokenizer, question_rows, args.batch_size)
    del question_model
    doc_ids, passage_rows, row_doc_indices, passage_stats = context_rows(
        documents, tokenizer, args.text_mode, args.max_length, args.overlap
    )
    passage_vectors = encode(context_model, tokenizer, passage_rows, args.batch_size)
    scores = score_passages(question_vectors, passage_vectors, similarity_mode)
    if len(passage_rows) != len(doc_ids):
        doc_scores = torch.full((len(query_ids), len(doc_ids)), -torch.inf, dtype=scores.dtype)
        columns = torch.tensor(row_doc_indices, dtype=torch.long).unsqueeze(0).expand(len(query_ids), -1)
        doc_scores.scatter_reduce_(1, columns, scores, reduce="amax", include_self=True)
        scores = doc_scores
    query_scores = torch.stack([scores[query_index[qid]] for qid in queries])
    summary, per_query = evaluate(PrecomputedRetriever(queries, doc_ids, query_scores), queries, relevance)
    summary["document_count"] = len(doc_ids)
    data_names = ["test.docs", "test.titles.queries", "test.2-1-0.qrel"]
    if args.text_mode != "preprocessed-single":
        data_names += ["raw/doc_dump.txt", "raw/nfdump.txt"]
    report = {
        "config": build_report_config(args, epoch, global_step, data_names, passage_stats, similarity_mode),
        "summary": summary,
        "per_query": per_query,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
