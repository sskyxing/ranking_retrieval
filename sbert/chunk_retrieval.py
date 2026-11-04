"""Exact document max-chunk retrieval and query-balanced, per-positive training."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


class ChunkIndex:
    def __init__(self, documents, tokenizer, max_length=256, overlap=64):
        capacity = max_length - tokenizer.num_special_tokens_to_add(pair=False)
        if capacity < 1 or overlap < 0 or overlap >= capacity:
            raise ValueError("Chunk overlap must be smaller than the content-token capacity")
        self.documents = {}
        self.max_length = max_length
        self.overlap = overlap
        for did, text in documents.items():
            tokens = tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
            chunks = []
            for start in range(0, max(1, len(tokens)), capacity - overlap):
                content = tokens[start:start + capacity]
                chunks.append(tokenizer.build_inputs_with_special_tokens(content))
                if start + capacity >= len(tokens):
                    break
            self.documents[did] = chunks

    def flatten(self, document_ids):
        rows, spans = [], []
        for did in document_ids:
            start = len(rows)
            rows.extend(self.documents[did])
            spans.append((start, len(rows)))
        return rows, spans


def features_from_ids(tokenizer, rows):
    return tokenizer.pad([{"input_ids": row, "attention_mask": [1] * len(row),
                           "token_type_ids": [0] * len(row)} for row in rows],
                          padding=True, return_tensors="pt", verbose=False)


def encode_rows(model, rows, batch_size):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            features = features_from_ids(model.tokenizer, rows[start:start + batch_size])
            output.append(model.sentence_embeddings(features).detach())
    if not output:
        raise ValueError("Cannot encode empty rows")
    return torch.cat(output)


def max_document_scores(query_embeddings, chunk_embeddings, spans):
    scores = F.normalize(query_embeddings, dim=1) @ F.normalize(chunk_embeddings, dim=1).T
    return torch.stack([scores[:, start:end].max(dim=1).values for start, end in spans], dim=1)


def per_positive_loss(scores, query_ids, document_ids, positive_ids, relevance,
                      total_positive_pairs, total_queries, temperature=0.05):
    """One CE per explicit positive; mask other known positives, weight queries equally."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if len(set(query_ids)) != len(query_ids) or len(set(document_ids)) != len(document_ids):
        raise ValueError("Duplicate batch query/document IDs")
    if scores.shape != (len(query_ids), len(document_ids)) or len(positive_ids) != len(query_ids):
        raise ValueError("Score and label dimensions differ")
    columns = {did: i for i, did in enumerate(document_ids)}
    targets, allowed, weights = [], [], []
    for qid, positive in zip(query_ids, positive_ids):
        positives = relevance[qid]
        if positive not in positives or positive not in columns:
            raise ValueError("Explicit positive is not a known training positive")
        targets.append(columns[positive])
        permitted = [did == positive or did not in positives for did in document_ids]
        if sum(permitted) < 2:
            raise ValueError("Query has no nonpositive candidate")
        allowed.append(permitted)
        weights.append(total_positive_pairs / (total_queries * len(positives)))
    mask = torch.tensor(allowed, device=scores.device, dtype=torch.bool)
    logits = (scores / temperature).masked_fill(~mask, -torch.inf)
    target_tensor = torch.tensor(targets, device=scores.device)
    losses = F.cross_entropy(logits, target_tensor, reduction="none")
    # Do not normalize by each batch's sum of weights: that would change the
    # intended global query-balanced objective under a full-positive schedule.
    loss = (losses * scores.new_tensor(weights)).mean()
    if not torch.isfinite(loss):
        raise ValueError("Non-finite per-positive retrieval loss")
    return loss


def cached_chunk_backward(model, index, query_texts, query_ids, document_ids, positive_ids,
                          relevance, total_positive_pairs, total_queries, micro_batch=8, temperature=0.05):
    """Exact chain-rule gradient caching with dropout disabled in both passes."""
    model.eval()  # eval disables dropout, but gradients still update the encoder.
    query_rows = model.tokenizer(query_texts, truncation=True, max_length=model.max_seq_length)["input_ids"]
    chunk_rows, spans = index.flatten(document_ids)
    query_cache = encode_rows(model, query_rows, micro_batch).requires_grad_(True)
    chunk_cache = encode_rows(model, chunk_rows, micro_batch).requires_grad_(True)
    scores = max_document_scores(query_cache, chunk_cache, spans)
    loss = per_positive_loss(scores, query_ids, document_ids, positive_ids, relevance,
                             total_positive_pairs, total_queries, temperature)
    loss.backward()
    # Recompute tiny encoder graphs and apply dL/dEmbedding immediately, avoiding
    # retention of an encoder graph for every document chunk in the batch.
    for rows, gradients in ((query_rows, query_cache.grad), (chunk_rows, chunk_cache.grad)):
        for start in range(0, len(rows), micro_batch):
            gradient = gradients[start:start + micro_batch]
            if not torch.count_nonzero(gradient):
                continue
            features = features_from_ids(model.tokenizer, rows[start:start + micro_batch])
            encoded = model.sentence_embeddings(features)
            torch.autograd.backward(encoded, gradient)
    return float(loss.detach()), len(chunk_rows)


class ChunkRetriever:
    def __init__(self, documents, queries, model, batch_size, overlap):
        self.doc_ids = sorted(documents)
        self.index = ChunkIndex(documents, model.tokenizer, model.max_seq_length, overlap)
        rows, self.spans = self.index.flatten(self.doc_ids)
        self.chunk_embeddings = F.normalize(encode_rows(model, rows, batch_size), dim=1)
        texts = list(dict.fromkeys(queries.values()))
        query_embeddings = F.normalize(model.encode(texts, batch_size), dim=1)
        self.queries = dict(zip(texts, query_embeddings))

    def search(self, query, k=10):
        # Document aggregation happens before top-k: one document occupies one slot.
        scores = max_document_scores(self.queries[query].unsqueeze(0), self.chunk_embeddings, self.spans)[0]
        ranking = torch.argsort(scores, descending=True, stable=True)[:k]
        return [self.doc_ids[i] for i in ranking.tolist()]
