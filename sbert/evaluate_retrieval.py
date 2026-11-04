"""Evaluate a locally trained SBERT encoder on NFCorpus using the BM25 metric code."""

import argparse
import json
from pathlib import Path
import sys


BASE = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE / "runs" / "nli-bert-base" / "sentence_transformer"
sys.path.insert(0, str(BASE.parent))
from bm25.run_bm25 import evaluate, load_qrels, load_texts  # noqa: E402


def load_metadata(model_dir):
    """Require an explicit local training export; never fall back to a Hub model."""
    if not model_dir.is_dir():
        raise ValueError(f"训练导出目录不存在: {model_dir}；请先完成训练或指定 --model-dir")
    metadata_path = model_dir / "sbert_config.json"
    if not metadata_path.is_file():
        raise ValueError(f"缺少训练导出信息: {metadata_path}；不能将预训练 BERT 当作复现结果")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("pooling") != "mean":
        raise ValueError("仅支持本次复现的 mean pooling 导出")
    if not metadata.get("training_provenance"):
        raise ValueError("sbert_config.json 缺少 training_provenance，无法确认训练来源")
    if not (model_dir / "config.json").is_file():
        raise ValueError(f"模型配置缺失: {model_dir / 'config.json'}")
    weight_names = ("model.safetensors", "model.safetensors.index.json",
                    "pytorch_model.bin", "pytorch_model.bin.index.json")
    if not any((model_dir / name).is_file() for name in weight_names):
        raise ValueError(f"模型权重缺失: {model_dir}")
    return metadata


class DenseRetriever:
    """Shared BERT + masked mean pooling + cosine exact search, entirely on CPU."""

    def __init__(self, documents, queries, model_dir, max_seq_length, batch_size, threads):
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer

        if not documents:
            raise ValueError("文档集合不能为空")
        self.torch = torch
        self.functional = functional
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.encoder = AutoModel.from_pretrained(str(model_dir), local_files_only=True)
        self.encoder.to(device="cpu", dtype=torch.float32).eval()
        model_limit = getattr(self.encoder.config, "max_position_embeddings", None)
        if model_limit is not None and max_seq_length > model_limit:
            raise ValueError(f"max_seq_length={max_seq_length} 超出模型位置上限 {model_limit}")
        # A stable descending sort of these lexically sorted IDs preserves the BM25 tie rule.
        self.doc_ids = sorted(documents)
        self.doc_embeddings = self.encode([documents[item_id] for item_id in self.doc_ids], "documents")
        query_texts = list(dict.fromkeys(queries.values()))
        query_embeddings = self.encode(query_texts, "queries")
        self.query_embeddings = dict(zip(query_texts, query_embeddings))
        self.cached_rankings = {}

    def encode(self, texts, description):
        torch = self.torch
        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                encoded = self.tokenizer(
                    texts[start:start + self.batch_size], padding=True, truncation=True,
                    max_length=self.max_seq_length, return_tensors="pt",
                )
                hidden = self.encoder(**encoded).last_hidden_state
                # The original pooling uses all non-padding positions, including CLS/SEP.
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                normalized = self.functional.normalize(pooled, p=2, dim=1)
                if not torch.isfinite(normalized).all():
                    raise ValueError(f"{description} 编码得到非有限数")
                batches.append(normalized)
                completed = min(start + self.batch_size, len(texts))
                if start == 0 or completed == len(texts) or (start // self.batch_size + 1) % 25 == 0:
                    print(f"Encoded {description}: {completed}/{len(texts)}", flush=True)
        if not batches:
            raise ValueError(f"{description} 不能为空")
        return torch.cat(batches)

    def search(self, query, k=10):
        if query not in self.cached_rankings:
            with self.torch.inference_mode():
                scores = self.doc_embeddings @ self.query_embeddings[query]
                ranking = self.torch.argsort(scores, descending=True, stable=True)
                # NFCorpus evaluation only needs the top 10, but retain the exact full ranking.
                self.cached_rankings[query] = (ranking.tolist(), scores.tolist())
        ranking, _ = self.cached_rankings[query]
        return [self.doc_ids[index] for index in ranking[:k]]

    def top_scores(self, query, k=10):
        self.search(query, k)
        ranking, scores = self.cached_rankings[query]
        return [scores[index] for index in ranking[:k]]


def main():
    # Windows terminals can otherwise default to a code page without Chinese glyphs.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=BASE.parent / "nfcorpus")
    parser.add_argument("--output", type=Path, default=BASE / "results.json")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=None,
                        help="包含特殊符号的总 token 数；默认读取训练导出配置")
    args = parser.parse_args()
    metadata = load_metadata(args.model_dir)
    if metadata.get("chunk_overlap") is not None:
        parser.error("This is a max-chunk model. Use evaluate_chunk_retrieval.py to preserve raw-text/document aggregation settings.")
    max_seq_length = (args.max_seq_length if args.max_seq_length is not None
                      else metadata.get("max_seq_length", 130))
    if not isinstance(max_seq_length, int):
        parser.error("训练配置中的 max_seq_length 必须为整数")
    if any(value <= 0 for value in (args.threads, args.batch_size, max_seq_length)):
        parser.error("threads、batch-size 和 max-seq-length 必须为正整数")
    if max_seq_length < 2:
        parser.error("max-seq-length 至少为 2（需要 CLS 和 SEP）")

    documents = load_texts(args.data_dir / "test.docs")
    queries = load_texts(args.data_dir / "test.titles.queries")
    relevant = load_qrels(args.data_dir / "test.2-1-0.qrel")
    missing_docs = set().union(*relevant.values()) - documents.keys()
    missing_queries = relevant.keys() - queries.keys()
    if missing_docs or missing_queries:
        raise ValueError(
            f"标注引用缺失 ID: documents={sorted(missing_docs)}, queries={sorted(missing_queries)}"
        )
    retriever = DenseRetriever(
        documents, queries, args.model_dir, max_seq_length, args.batch_size, args.threads
    )
    summary, per_query = evaluate(retriever, queries, relevant)
    summary["document_count"] = len(documents)
    for query_id, result in per_query.items():
        result["top_10_scores"] = retriever.top_scores(queries[query_id])
    import torch
    import transformers
    report = {
        "config": {
            "documents": "test.docs",
            "queries": "test.titles.queries",
            "qrels": "test.2-1-0.qrel",
            "model_dir": str(args.model_dir.resolve()),
            "training_export": metadata,
            "device": "cpu",
            "dtype": "float32",
            "threads": args.threads,
            "batch_size": args.batch_size,
            "max_seq_length": max_seq_length,
            "pooling": "attention-masked mean; includes non-padding CLS/SEP",
            "normalization": "L2",
            "similarity": "dot product of normalized vectors (cosine)",
            "search": "exact; every document participates",
            "tie_break": "score descending, doc_id ascending",
            "ndcg": "linear relevance gains; log2(rank + 1) discount",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "summary": summary,
        "per_query": per_query,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Recall@10: {summary['recall_at_10']:.6f}")
    print(f"nDCG@10: {summary['ndcg_at_10']:.6f}")
    print(f"Documents: {len(documents)}; queries: {len(queries)}; "
          f"evaluated: {len(per_query)}; skipped: {summary['skipped_query_count']}")
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
