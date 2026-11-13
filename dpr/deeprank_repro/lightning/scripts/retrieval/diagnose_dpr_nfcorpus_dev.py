"""Compare DPR dot-product and cosine retrieval on NFCorpus dev only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from evaluate_dpr_nfcorpus import PrecomputedRetriever, context_rows, encode, load_encoders, sha256


def load_dev_protocol(data_dir: Path):
    """Use the same raw-text loader and metric implementation as the test report."""

    baseline_repo = data_dir.resolve().parent
    if not (baseline_repo / "bm25" / "run_bm25.py").is_file():
        raise FileNotFoundError(f"BM25 metric implementation is missing under {baseline_repo}")
    if not (baseline_repo / "sbert" / "retrieval_data.py").is_file():
        raise FileNotFoundError(f"SBERT raw-text loader is missing under {baseline_repo}")
    sys.path.insert(0, str(baseline_repo))
    from bm25.run_bm25 import evaluate
    from sbert.retrieval_data import load_split

    documents, queries, relevance = load_split(data_dir, "dev", raw_text=True)
    if (len(documents), len(queries), len(relevance)) != (3193, 325, 324):
        raise ValueError("Unexpected NFCorpus dev split; confirm the supplied baseline data.")
    return documents, queries, relevance, evaluate


def aggregate_document_scores(query_vectors, passage_vectors, passage_doc_indices, doc_count):
    """Score each document by the maximum score of its overlapping chunks."""

    if query_vectors.ndim != 2 or passage_vectors.ndim != 2 or query_vectors.shape[1] != passage_vectors.shape[1]:
        raise ValueError("Question and passage vectors must have matching embedding dimensions.")
    if len(passage_doc_indices) != len(passage_vectors):
        raise ValueError("Every passage needs a document index.")
    if doc_count < 1 or len(passage_vectors) == 0:
        raise ValueError("The development corpus cannot be empty.")
    indices = torch.as_tensor(passage_doc_indices, dtype=torch.long, device=passage_vectors.device)
    if indices.min() < 0 or indices.max() >= doc_count:
        raise ValueError("A passage document index is outside the corpus.")
    scores = query_vectors @ passage_vectors.T
    document_scores = scores.new_full((len(query_vectors), doc_count), -torch.inf)
    columns = indices.unsqueeze(0).expand(len(query_vectors), -1)
    return document_scores.scatter_reduce_(1, columns, scores, reduce="amax", include_self=True)


def compare_scoring(query_vectors, passage_vectors, passage_doc_indices, doc_count):
    """Reuse the same embeddings and document chunks for both scoring rules."""

    question_norms = torch.linalg.vector_norm(query_vectors, dim=1)
    passage_norms = torch.linalg.vector_norm(passage_vectors, dim=1)
    if not torch.isfinite(question_norms).all() or not torch.isfinite(passage_norms).all():
        raise ValueError("DPR embeddings have nonfinite norms.")
    if (question_norms <= 0).any() or (passage_norms <= 0).any():
        raise ValueError("Cosine scoring requires nonzero DPR embeddings.")
    dot = aggregate_document_scores(query_vectors, passage_vectors, passage_doc_indices, doc_count)
    cosine = aggregate_document_scores(
        F.normalize(query_vectors, dim=1), F.normalize(passage_vectors, dim=1), passage_doc_indices, doc_count
    )
    return {"dot": dot, "cosine": cosine}, question_norms, passage_norms


def summarize(values):
    values = values.float()
    return {
        "count": len(values),
        "min": float(values.min()),
        "p10": float(torch.quantile(values, 0.1)),
        "median": float(torch.median(values)),
        "p90": float(torch.quantile(values, 0.9)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def pearson(left, right):
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float((left @ right) / denominator) if denominator > 0 else None


def summarize_bias(scores_by_mode, question_norms, passage_norms, passage_doc_indices, doc_count):
    """Describe norm and chunk-count associations; these are not causal estimates."""

    indices = torch.as_tensor(passage_doc_indices, dtype=torch.long)
    counts = torch.bincount(indices, minlength=doc_count)
    max_norms = torch.zeros(doc_count, dtype=passage_norms.dtype)
    max_norms.scatter_reduce_(0, indices, passage_norms, reduce="amax", include_self=True)
    per_mode = {}
    top_indices = {}
    for mode, scores in scores_by_mode.items():
        top = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :10]
        top_indices[mode] = top
        frequency = torch.bincount(top.flatten(), minlength=doc_count)
        per_mode[mode] = {
            "top_10_mean_document_chunk_count": float(counts[top].float().mean()),
            "top_10_mean_document_max_passage_norm": float(max_norms[top].mean()),
            "retrieval_frequency_vs_chunk_count_pearson": pearson(frequency, counts),
            "retrieval_frequency_vs_max_passage_norm_pearson": pearson(frequency, max_norms),
        }
    overlap = [
        len(set(dot.tolist()) & set(cosine.tolist()))
        for dot, cosine in zip(top_indices["dot"], top_indices["cosine"])
    ]
    return {
        "question_norm": summarize(question_norms),
        "passage_norm": summarize(passage_norms),
        "document_chunk_count": summarize(counts),
        "document_max_passage_norm": summarize(max_norms),
        "mean_dot_cosine_top_10_overlap": sum(overlap) / len(overlap),
        "ranking_bias": per_mode,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--source-aml-run-id", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1 or not 8 <= args.max_length <= 512:
        parser.error("batch-size and threads must be positive; max-length must be 8..512")
    if not 0 <= args.overlap < args.max_length - 3:
        parser.error("overlap must be between zero and max-length - 4")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    documents, queries, relevance, evaluate = load_dev_protocol(args.data_dir)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer, do_lower_case=True)
    question_model, passage_model, epoch, global_step = load_encoders(args.checkpoint)
    query_ids = sorted(queries)
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
    doc_ids, rows, row_doc_indices, passage_stats = context_rows(
        documents, tokenizer, "raw-chunks", args.max_length, args.overlap
    )
    passage_vectors = encode(passage_model, tokenizer, rows, args.batch_size)
    scores_by_mode, question_norms, passage_norms = compare_scoring(
        question_vectors, passage_vectors, row_doc_indices, len(doc_ids)
    )
    query_index = {qid: index for index, qid in enumerate(query_ids)}
    results = {}
    for mode, scores in scores_by_mode.items():
        original_order = torch.stack([scores[query_index[qid]] for qid in queries])
        summary, per_query = evaluate(PrecomputedRetriever(queries, doc_ids, original_order), queries, relevance)
        summary["document_count"] = len(doc_ids)
        results[mode] = {"summary": summary, "per_query": per_query}
    diagnostics = summarize_bias(scores_by_mode, question_norms, passage_norms, row_doc_indices, len(doc_ids))
    comparable_ids = set(results["dot"]["per_query"]) & set(results["cosine"]["per_query"])
    diagnostics["cosine_vs_dot_query_count"] = {
        metric: {
            "better": sum(
                results["cosine"]["per_query"][qid][metric] > results["dot"]["per_query"][qid][metric]
                for qid in comparable_ids
            ),
            "tie": sum(
                results["cosine"]["per_query"][qid][metric] == results["dot"]["per_query"][qid][metric]
                for qid in comparable_ids
            ),
            "worse": sum(
                results["cosine"]["per_query"][qid][metric] < results["dot"]["per_query"][qid][metric]
                for qid in comparable_ids
            ),
        }
        for metric in ("recall_at_10", "ndcg_at_10")
    }
    report = {
        "config": {
            "split": "dev",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_bytes": args.checkpoint.stat().st_size,
            "checkpoint_epoch": epoch,
            "checkpoint_global_step": global_step,
            "source_aml_run_id": args.source_aml_run_id,
            "tokenizer": args.tokenizer,
            "text_mode": "raw-chunks",
            "max_length": args.max_length,
            "overlap": args.overlap,
            "document_score": "maximum score across document chunks, separately for dot and cosine",
            "tie_break": "score descending, doc_id ascending",
            "data_sha256": {
                name: sha256(args.data_dir / name)
                for name in (
                    "dev.docs", "dev.titles.queries", "dev.2-1-0.qrel",
                    "raw/doc_dump.txt", "raw/nfdump.txt",
                )
            },
            **passage_stats,
        },
        "scoring": results,
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({mode: result["summary"] for mode, result in results.items()}, indent=2))
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
