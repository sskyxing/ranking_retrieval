"""仅用 Python 标准库，在 NFCorpus 测试集上运行 BM25、Recall@10 和 nDCG@10。"""

import argparse
import heapq
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


def tokenize(text):
    """统一对文档和查询做小写、英文/数字分词，不做词干化。"""
    return re.findall(r"[a-z0-9]+", text.lower())


def load_texts(path):
    """读取 ID<TAB>TEXT 格式，保留原始文档/查询 ID。"""
    texts = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item_id, text = line.rstrip("\r\n").split("\t", 1)
            if item_id in texts:
                raise ValueError(f"重复 ID: {item_id} ({path})")
            texts[item_id] = text
    return texts


def load_qrels(path):
    """保留 TREC 四列标注中的正相关度等级，供 Recall 和 nDCG 使用。"""
    relevant = defaultdict(dict)
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            query_id, _, doc_id, relevance = line.split()
            if int(relevance) > 0:
                relevant[query_id][doc_id] = int(relevance)
    return relevant


class BM25:
    def __init__(self, documents, k1=1.2, b=0.75):
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 必须是有限正数")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b 必须是 [0, 1] 内的有限数")
        if not documents:
            raise ValueError("文档集合不能为空")
        self.doc_ids = sorted(documents)
        self.k1 = k1
        self.postings = defaultdict(list)
        lengths = []
        # 倒排表存储每个词出现在哪些文档，以及各文档内的词频。
        for index, doc_id in enumerate(self.doc_ids):
            counts = Counter(tokenize(documents[doc_id]))
            lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((index, frequency))
        self.avgdl = sum(lengths) / len(lengths)
        if self.avgdl == 0:
            raise ValueError("所有文档分词后均为空")
        self.norm = [k1 * (1 - b + b * n / self.avgdl) for n in lengths]
        # 非负 IDF；df 是包含该词的文档数量。
        self.idf = {
            term: math.log1p((len(lengths) - len(rows) + 0.5) / (len(rows) + 0.5))
            for term, rows in self.postings.items()
        }

    def search(self, query, k=10):
        scores = [0.0] * len(self.doc_ids)
        # 同一查询词仅计一次，不额外引入查询词频权重；排序保证累加顺序固定。
        for term in sorted(set(tokenize(query))):
            for index, frequency in self.postings.get(term, ()):
                scores[index] += (
                    self.idf[term] * frequency * (self.k1 + 1)
                    / (frequency + self.norm[index])
                )
        # 所有文档参与排名：分数降序，同分按文档 ID 升序，零分也可补齐。
        best = heapq.nsmallest(
            k, range(len(scores)), key=lambda i: (-scores[i], self.doc_ids[i])
        )
        return [self.doc_ids[index] for index in best]


def ndcg_at_k(retrieved, relevance, k=10):
    """使用线性增益（原始相关度等级），与 trec_eval 的 ndcg_cut 口径一致。"""
    dcg = sum(
        relevance.get(doc_id, 0) / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], start=1)
    )
    # 理想排名取该查询全部正例中相关度最高的 k 篇，不局限于检索结果。
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def evaluate(model, queries, relevant):
    results = {}
    skipped = []
    for query_id, query in queries.items():
        positives = relevant.get(query_id, {})
        if not positives:
            skipped.append(query_id)
            continue
        retrieved = model.search(query, k=10)
        hits = len(set(retrieved).intersection(positives))
        results[query_id] = {
            "hits_at_10": hits,
            "relevant_count": len(positives),
            "recall_at_10": hits / len(positives),
            "ndcg_at_10": ndcg_at_k(retrieved, positives),
            "top_10_doc_ids": retrieved,
        }
    if not results:
        raise ValueError("没有可评估的查询：每个查询都缺少正例")
    return {
        "recall_at_10": sum(r["recall_at_10"] for r in results.values()) / len(results),
        "ndcg_at_10": sum(r["ndcg_at_10"] for r in results.values()) / len(results),
        "query_count": len(queries),
        "evaluated_query_count": len(results),
        "skipped_query_count": len(skipped),
        "skipped_query_ids": skipped,
    }, results


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=base.parent / "nfcorpus")
    parser.add_argument("--output", type=Path, default=base / "results.json")
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()
    documents = load_texts(args.data_dir / "test.docs")
    queries = load_texts(args.data_dir / "test.titles.queries")
    relevant = load_qrels(args.data_dir / "test.2-1-0.qrel")
    # 缺失正例应报错，不能静默缩小 Recall 的分母。
    missing_docs = set().union(*relevant.values()) - documents.keys()
    missing_queries = relevant.keys() - queries.keys()
    if missing_docs or missing_queries:
        raise ValueError(
            f"标注引用缺失 ID: documents={sorted(missing_docs)}, queries={sorted(missing_queries)}"
        )
    model = BM25(documents, k1=args.k1, b=args.b)
    summary, per_query = evaluate(model, queries, relevant)
    summary["document_count"] = len(documents)
    report = {
        "config": {
            "documents": "test.docs",
            "queries": "test.titles.queries",
            "qrels": "test.2-1-0.qrel",
            "k1": args.k1, "b": args.b,
            "tokenizer": "lowercase + [a-z0-9]+; no stemming; unique query terms",
            "idf": "ln(1 + (N - df + 0.5) / (df + 0.5))",
            "tie_break": "score descending, doc_id ascending",
            "ndcg": "linear relevance gains; log2(rank + 1) discount",
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
