"""NFCorpus train-only hard negatives and a multi-positive retrieval objective."""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from bm25.run_bm25 import BM25, load_texts  # noqa: E402
try:
    from .train import atomic_json, sha256
except ImportError:
    from train import atomic_json, sha256


def load_raw_texts(data_dir, documents, queries):
    """Map only requested split IDs to original titles/abstracts; never read raw labels."""
    raw_documents, raw_queries = {}, {}
    sources = (("doc_dump.txt", documents, raw_documents, True),
               ("nfdump.txt", queries, raw_queries, False))
    for filename, requested, target, is_document in sources:
        with (Path(data_dir) / "raw" / filename).open(encoding="utf-8") as stream:
            for line in stream:
                fields = line.rstrip("\r\n").split("\t", 3)
                if fields[0] not in requested:
                    continue
                if len(fields) != 4 or fields[0] in target:
                    raise ValueError(f"Invalid/duplicate raw record in {filename}: {fields[0]}")
                text = (fields[2].strip() + "\n" + fields[3].strip()) if is_document else fields[2].strip()
                if not text.strip():
                    raise ValueError(f"Empty raw text: {fields[0]}")
                target[fields[0]] = text
        if target.keys() != requested.keys():
            raise ValueError(f"Missing raw IDs in {filename}: {sorted(requested.keys() - target.keys())[:10]}")
    return raw_documents, raw_queries


def load_split(data_dir: Path, split: str, raw_text=False):
    if split not in ("train", "dev", "test"):
        raise ValueError(f"Unsupported split: {split}")
    documents = load_texts(data_dir / f"{split}.docs")
    queries = load_texts(data_dir / f"{split}.titles.queries")
    relevance = {}
    if not documents or not queries:
        raise ValueError(f"Empty {split} documents/queries")
    with (data_dir / f"{split}.2-1-0.qrel").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            qid, _, did, grade = line.split()
            grade = int(grade)
            if qid not in queries or did not in documents or grade not in (0, 1, 2):
                raise ValueError(f"Invalid {split} qrel: {line!r}")
            if grade > 0:
                relevance.setdefault(qid, {})[did] = grade
    if not relevance:
        raise ValueError(f"No positive {split} judgments")
    if raw_text:
        documents, queries = load_raw_texts(data_dir, documents, queries)
    return documents, queries, relevance


def prepare_training_data(data_dir: Path, cache_path: Path, hard_negative_count=20, raw_text=False):
    """Read train/dev only. Negatives never use dev or test judgments."""
    if hard_negative_count < 1:
        raise ValueError("hard_negative_count must be positive")
    data_dir, cache_path = Path(data_dir).resolve(), Path(cache_path)
    hashes = {f"{split}.{suffix}": sha256(data_dir / f"{split}.{suffix}")
              for split in ("train", "dev")
              for suffix in ("docs", "titles.queries", "2-1-0.qrel")}
    if raw_text:
        hashes.update({f"raw/{name}": sha256(data_dir / "raw" / name)
                       for name in ("doc_dump.txt", "nfdump.txt")})
    train_docs, train_queries, relevance = load_split(data_dir, "train", raw_text)
    dev_docs, dev_queries, dev_relevance = load_split(data_dir, "dev", raw_text)
    normalize = lambda text: " ".join(text.lower().split())
    dev_texts = {normalize(text) for text in dev_queries.values()}
    groups = {}
    excluded = []
    for qid, text in sorted(train_queries.items()):
        if normalize(text) in dev_texts:
            excluded.append(qid)
        else:
            groups.setdefault(normalize(text), []).append(qid)
    valid, aliases = [], {}
    merged_relevance = dict(relevance)
    for ids in groups.values():
        positives = {}
        for qid in ids:
            for did, grade in relevance.get(qid, {}).items():
                positives[did] = max(positives.get(did, 0), grade)
        if positives:
            representative = ids[0]
            valid.append(representative)
            aliases[representative] = ids
            merged_relevance[representative] = positives
    valid.sort()
    if not valid:
        raise ValueError("No training queries remain after train/dev overlap filtering")
    parameters = {"schema_version": 1, "source_sha256": hashes,
                  "hard_negative_count": hard_negative_count, "bm25": {"k1": 1.2, "b": 0.75},
                  "exclude_dev_query_text": "lowercase + collapse whitespace",
                  "train_query_duplicates": "merge known positives; one representative"}
    if raw_text:
        parameters.update({"schema_version": 2, "text_source": "raw title + abstract; raw query title"})
    expected_hash = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
    cached = None
    if cache_path.is_file():
        try:
            candidate = json.loads(cache_path.read_text(encoding="utf-8"))
            if candidate.get("manifest", {}).get("data_hash") == expected_hash:
                cached = candidate
        except (ValueError, KeyError):
            pass
    if cached is None:
        print(f"Mining BM25 hard negatives for {len(valid)} train queries", flush=True)
        bm25 = BM25(train_docs)
        negatives = {}
        for index, qid in enumerate(valid):
            positives = merged_relevance[qid]
            # At most len(positives) results can be filtered, so this depth suffices.
            ranking = bm25.search(train_queries[qid], min(len(train_docs), len(positives) + hard_negative_count))
            negatives[qid] = [did for did in ranking if did not in positives][:hard_negative_count]
            if not negatives[qid]:
                raise ValueError(f"No negative candidates for query {qid}")
            if (index + 1) % 500 == 0 or index + 1 == len(valid):
                print(f"Mined {index + 1}/{len(valid)} queries", flush=True)
        cached = {"manifest": {**parameters, "data_hash": expected_hash,
                  "excluded_dev_overlap_query_ids": excluded,
                  "query_aliases": aliases,
                  "negative_note": "BM25 nonpositive candidates include unjudged documents; not verified negatives",
                  "counts": {"train_documents": len(train_docs), "train_queries": len(train_queries),
                             "train_positive_pairs": sum(len(v) for v in relevance.values()),
                             "eligible_train_queries": len(valid), "excluded_dev_overlap": len(excluded),
                             "dev_documents": len(dev_docs), "dev_queries": len(dev_queries)}},
                  "hard_negatives": negatives}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(cache_path, cached)
    negatives = cached["hard_negatives"]
    if set(negatives) != set(valid):
        raise ValueError("Cached hard negative query IDs differ from current data")
    for qid, candidates in negatives.items():
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError(f"Empty or duplicate cached negatives: {qid}")
        if any(did not in train_docs or did in merged_relevance[qid] for did in candidates):
            raise ValueError(f"Invalid or known-positive hard negative: {qid}")
    cached["manifest"]["hard_negatives_sha256"] = hashlib.sha256(
        json.dumps(negatives, sort_keys=True).encode()).hexdigest()
    return {"train_documents": train_docs, "train_queries": train_queries,
            "train_relevance": merged_relevance, "dev_documents": dev_docs,
            "dev_queries": dev_queries, "dev_relevance": dev_relevance,
            "valid_query_ids": valid, "hard_negatives": negatives, "manifest": cached["manifest"]}


def build_epoch_examples(data, epoch: int, seed: int):
    generator = random.Random(seed + epoch)
    qids = list(data["valid_query_ids"])
    generator.shuffle(qids)
    return [(qid, generator.choice(sorted(data["train_relevance"][qid])),
             generator.choice(data["hard_negatives"][qid])) for qid in qids]


def build_epoch_batches(data, epoch: int, seed: int, batch_size: int, sampling="query"):
    """Deterministic batches: unique queries, with optional exact full positive coverage."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if sampling == "query":
        examples = build_epoch_examples(data, epoch, seed)
        return [examples[start:start + batch_size] for start in range(0, len(examples), batch_size)]
    if sampling != "all-positives":
        raise ValueError(f"Unknown sampling mode: {sampling}")
    generator = random.Random(seed + epoch)
    queues, heap = {}, []
    for qid in sorted(data["valid_query_ids"]):
        positives = sorted(data["train_relevance"][qid])
        negatives = list(data["hard_negatives"][qid])
        if not positives or not negatives:
            raise ValueError(f"Empty positive/negative pool: {qid}")
        generator.shuffle(positives)
        generator.shuffle(negatives)
        # Random pair priorities interleave all queries across the epoch. Delayed
        # reinsertion guarantees a query occurs at most once inside each batch.
        priorities = sorted(generator.random() for _ in positives)
        queues[qid] = (positives, negatives, priorities)
        heapq.heappush(heap, (priorities[0], qid, 0))
    batches = []
    while heap:
        batch, pending = [], []
        for _ in range(min(batch_size, len(heap))):
            _, qid, cursor = heapq.heappop(heap)
            positives, negatives, priorities = queues[qid]
            batch.append((qid, positives[cursor], negatives[cursor % len(negatives)]))
            if cursor + 1 < len(positives):
                pending.append((priorities[cursor + 1], qid, cursor + 1))
        for item in pending:
            heapq.heappush(heap, item)
        batches.append(batch)
    expected = {(qid, did) for qid in data["valid_query_ids"] for did in data["train_relevance"][qid]}
    actual = [(qid, positive) for batch in batches for qid, positive, _ in batch]
    if len(actual) != len(expected) or set(actual) != expected:
        raise AssertionError("Full-positive schedule omitted or duplicated training pairs")
    return batches


def contrastive_loss(query_embeddings, document_embeddings, query_ids, document_ids, relevance, temperature=0.05):
    """Multi-positive query->document InfoNCE, avoiding known in-batch false negatives."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if len(set(query_ids)) != len(query_ids) or len(set(document_ids)) != len(document_ids):
        raise ValueError("Batch query/document IDs must each be unique")
    if query_embeddings.shape[0] != len(query_ids) or document_embeddings.shape[0] != len(document_ids):
        raise ValueError("Embedding rows must match IDs")
    logits = F.normalize(query_embeddings, dim=1) @ F.normalize(document_embeddings, dim=1).T / temperature
    positive = torch.tensor([[relevance.get(qid, {}).get(did, 0) > 0 for did in document_ids]
                             for qid in query_ids], dtype=torch.bool, device=logits.device)
    if not positive.any(dim=1).all() or positive.all(dim=1).any():
        raise ValueError("Each query must have at least one known positive and one nonpositive candidate")
    loss = (torch.logsumexp(logits, dim=1) -
            torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)).mean()
    if not torch.isfinite(loss):
        raise ValueError("Non-finite retrieval loss")
    return loss
