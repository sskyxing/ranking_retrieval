"""NFCorpus train/dev data with document-aware DPR chunks and labels."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


def load_texts(path: Path) -> Dict[str, str]:
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item_id, text = line.rstrip("\r\n").split("\t", 1)
            if item_id in result:
                raise ValueError(f"Duplicate ID in {path}: {item_id}")
            result[item_id] = text
    return result


def load_qrels(path: Path, queries: Dict[str, str], documents: Dict[str, str]) -> Dict[str, Dict[str, int]]:
    relevance = defaultdict(dict)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            query_id, _, doc_id, grade_string = line.split()
            grade = int(grade_string)
            if query_id not in queries or doc_id not in documents or grade not in (0, 1, 2):
                raise ValueError(f"Invalid qrel in {path}: {line!r}")
            if grade > 0:
                relevance[query_id][doc_id] = grade
    return dict(relevance)


def load_raw_texts(root: Path, documents: Dict[str, str], queries: Dict[str, str]):
    raw_docs, raw_queries = {}, {}
    for filename, requested, target, is_doc in (
        ("doc_dump.txt", documents, raw_docs, True),
        ("nfdump.txt", queries, raw_queries, False),
    ):
        with (root / "raw" / filename).open(encoding="utf-8") as stream:
            for line in stream:
                fields = line.rstrip("\r\n").split("\t", 3)
                if fields[0] not in requested:
                    continue
                if len(fields) != 4 or fields[0] in target:
                    raise ValueError(f"Invalid or duplicate raw record: {filename} {fields[0]}")
                text = (fields[2].strip(), fields[3].strip()) if is_doc else fields[2].strip()
                if (is_doc and not any(text)) or (not is_doc and not text):
                    raise ValueError(f"Empty raw text: {fields[0]}")
                target[fields[0]] = text
        if target.keys() != requested.keys():
            raise ValueError(f"Missing raw IDs in {filename}: {sorted(requested.keys() - target.keys())[:10]}")
    return raw_docs, raw_queries


def load_split(root: Path, split: str):
    documents = load_texts(root / f"{split}.docs")
    queries = load_texts(root / f"{split}.titles.queries")
    relevance = load_qrels(root / f"{split}.2-1-0.qrel", queries, documents)
    raw_docs, raw_queries = load_raw_texts(root, documents, queries)
    return raw_docs, raw_queries, relevance


def normalize_query(text: str) -> str:
    return " ".join(text.lower().split())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_train_split(root: Path):
    """Use only train qrels plus dev query texts for duplicate exclusion."""

    train_docs = load_texts(root / "train.docs")
    train_queries = load_texts(root / "train.titles.queries")
    train_relevance = load_qrels(root / "train.2-1-0.qrel", train_queries, train_docs)
    dev_query_ids = load_texts(root / "dev.titles.queries")
    raw_docs, raw_queries = load_raw_texts(root, train_docs, train_queries)
    _, raw_dev_queries = load_raw_texts(root, {}, dev_query_ids)
    query_texts, positives, aliases = merged_training_queries(raw_queries, train_relevance, raw_dev_queries)
    return raw_docs, query_texts, positives, aliases


def merged_training_queries(train_queries, train_relevance, dev_queries):
    """Merge duplicate train titles and drop titles duplicated in dev."""

    dev_texts = {normalize_query(text) for text in dev_queries.values()}
    groups = defaultdict(list)
    for query_id in sorted(train_queries):
        normalized = normalize_query(train_queries[query_id])
        if normalized not in dev_texts:
            groups[normalized].append(query_id)
    query_texts, positives, aliases = {}, {}, {}
    for ids in groups.values():
        combined = {}
        for query_id in ids:
            for doc_id, grade in train_relevance.get(query_id, {}).items():
                combined[doc_id] = max(combined.get(doc_id, 0), grade)
        if combined:
            representative = ids[0]
            query_texts[representative] = train_queries[representative]
            positives[representative] = combined
            aliases[representative] = ids
    if not positives:
        raise ValueError("NFCorpus train has no eligible positive query/document pairs.")
    return query_texts, positives, aliases


class NFCorpusTrainDataset(Dataset):
    """Every known positive pair occurs once per epoch, with a train-only BM25 negative."""

    def __init__(self, queries, positives, hard_negatives, seed: int):
        self.queries = queries
        self.positives = positives
        self.hard_negatives = hard_negatives
        self.seed = seed
        self.epoch = 0
        self.pairs = self.epoch_pairs(0)
        if not self.pairs:
            raise ValueError("NFCorpus training pairs are empty.")

    def epoch_pairs(self, epoch: int):
        rng = random.Random(self.seed + epoch)
        queues = {}
        for qid in sorted(self.positives):
            positives = sorted(self.positives[qid])
            rng.shuffle(positives)
            queues[qid] = positives
        active = sorted(queues)
        rng.shuffle(active)
        result = []
        while active:
            next_active = []
            for qid in active:
                result.append((qid, queues[qid].pop()))
                if queues[qid]:
                    next_active.append(qid)
            rng.shuffle(next_active)
            active = next_active
        return result

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.pairs = self.epoch_pairs(epoch)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        qid, positive = self.pairs[index]
        negatives = self.hard_negatives[qid]
        rng = random.Random(self.seed + 1000003 * self.epoch + index)
        negative = negatives[rng.randrange(len(negatives))]
        return qid, self.queries[qid], positive, negative


class NFCorpusDevDataset(Dataset):
    def __init__(self, queries):
        self.items = sorted(queries.items())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class NFCorpusCollator:
    def __init__(
        self,
        tokenizer,
        documents,
        max_question_length,
        max_context_length,
        overlap,
        training,
        query_to_index=None,
        doc_to_index=None,
    ):
        self.tokenizer = tokenizer
        self.documents = documents
        self.max_question_length = max_question_length
        self.max_context_length = max_context_length
        self.overlap = overlap
        self.training = training
        self.chunk_cache = {}
        self.query_to_index = query_to_index
        self.doc_to_index = doc_to_index

    def chunks(self, doc_id):
        if doc_id not in self.chunk_cache:
            title, abstract = self.documents[doc_id]
            title_ids = self.tokenizer.encode(title, add_special_tokens=False, verbose=False)[
                : self.max_context_length - 4
            ]
            abstract_ids = self.tokenizer.encode(abstract, add_special_tokens=False, verbose=False)
            capacity = self.max_context_length - len(title_ids) - 3
            stride = capacity - min(self.overlap, capacity - 1)
            rows = []
            for start in range(0, max(1, len(abstract_ids)), stride):
                rows.append(
                    self.tokenizer.build_inputs_with_special_tokens(title_ids, abstract_ids[start : start + capacity])
                )
                if start + capacity >= len(abstract_ids):
                    break
            self.chunk_cache[doc_id] = rows
        return self.chunk_cache[doc_id]

    def tokenize_questions(self, texts):
        encoded = self.tokenizer(
            [text[:-1] if text.endswith("?") else text for text in texts],
            padding="max_length",
            truncation=True,
            max_length=self.max_question_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "token_type_ids": torch.zeros_like(encoded["input_ids"]),
        }

    def tokenize_chunks(self, rows):
        encoded = self.tokenizer.pad(
            [{"input_ids": row, "attention_mask": [1] * len(row)} for row in rows],
            padding="max_length",
            max_length=self.max_context_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "token_type_ids": torch.zeros_like(encoded["input_ids"]),
        }

    def __call__(self, items):
        if not items:
            raise ValueError("Cannot collate an empty NFCorpus batch.")
        if self.training:
            query_ids, questions, positives, negatives = zip(*items)
            document_ids = sorted(set(positives) | set(negatives))
            chunks, chunk_doc_indices = [], []
            for doc_index, doc_id in enumerate(document_ids):
                rows = self.chunks(doc_id)
                chunks.extend(rows)
                chunk_doc_indices.extend([doc_index] * len(rows))
        else:
            query_ids, questions = zip(*items)
        batch = {
            "query_ids": list(query_ids),
            "question": self.tokenize_questions(questions),
        }
        if self.training:
            batch["context"] = self.tokenize_chunks(chunks)
            batch["query_indices"] = torch.tensor([self.query_to_index[qid] for qid in query_ids], dtype=torch.long)
            batch["target_doc_indices"] = torch.tensor([self.doc_to_index[did] for did in positives], dtype=torch.long)
            batch["chunk_global_doc_indices"] = torch.tensor(
                [self.doc_to_index[document_ids[index]] for index in chunk_doc_indices], dtype=torch.long
            )
        return batch


class NFCorpusDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        tokenizer_name_or_path: str = "bert-base-uncased",
        use_fast_tokenizer: bool = False,
        batch_size: int = 8,
        val_batch_size: int = 16,
        num_workers: int = 0,
        max_question_length: int = 256,
        max_context_length: int = 256,
        context_overlap: int = 64,
        hard_negative_count: int = 20,
        hard_negatives_per_query: int = 1,
        seed: int = 42,
        drop_last: bool = False,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ):
        super().__init__()
        if batch_size < 1 or val_batch_size < 1 or num_workers < 0 or hard_negative_count < 1:
            raise ValueError("Invalid NFCorpus batch/worker/negative counts.")
        if hard_negatives_per_query != 1:
            raise ValueError("Current NFCorpus collator supports one hard negative per positive pair.")
        if persistent_workers:
            raise ValueError("NFCorpus epoch schedules require persistent_workers=false.")
        if max_context_length < 8 or context_overlap < 0 or context_overlap >= max_context_length - 3:
            raise ValueError("Invalid NFCorpus context chunk configuration.")
        self.save_hyperparameters()
        self.data_root = Path(data_root)
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.use_fast_tokenizer = use_fast_tokenizer
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.max_question_length = max_question_length
        self.max_context_length = max_context_length
        self.context_overlap = context_overlap
        self.hard_negative_count = hard_negative_count
        self.seed = seed
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.persistent_workers = False
        self.tokenizer = None
        self.train_dataset = None
        self.dev_dataset = None
        self.dev_query_id_to_index = None

    def prepare_data(self):
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, use_fast=self.use_fast_tokenizer)

    def setup(self, stage: Optional[str] = None):
        from transformers import AutoTokenizer

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_path, use_fast=self.use_fast_tokenizer
            )
        if stage not in (None, "fit", "validate"):
            return
        train_docs, query_texts, positives, aliases = load_train_split(self.data_root)
        dev_docs, dev_queries, dev_relevance = load_split(self.data_root, "dev")
        cache_path = self.data_root / "hard_negatives_raw_train.json"
        with cache_path.open(encoding="utf-8") as stream:
            cache = json.load(stream)
        if cache["manifest"]["hard_negative_count"] != self.hard_negative_count:
            raise ValueError("BM25 hard-negative cache count does not match the configuration.")
        if set(cache["hard_negatives"]) != set(query_texts):
            raise ValueError("BM25 hard-negative cache query IDs do not match filtered training queries.")
        if cache["manifest"].get("bm25") != {"k1": 1.2, "b": 0.75}:
            raise ValueError("BM25 hard-negative cache has unexpected retrieval parameters.")
        source_hashes = cache["manifest"].get("source_sha256", {})
        for name in (
            "train.docs",
            "train.titles.queries",
            "train.2-1-0.qrel",
            "dev.titles.queries",
            "raw/doc_dump.txt",
            "raw/nfdump.txt",
        ):
            if source_hashes.get(name) != sha256(self.data_root / name):
                raise ValueError(f"BM25 hard-negative cache has a stale or missing source: {name}.")
        hard_negatives = cache["hard_negatives"]
        for qid, candidates in hard_negatives.items():
            if not candidates or len(candidates) != len(set(candidates)):
                raise ValueError(f"Empty or duplicate BM25 negatives for {qid}")
            if any(did not in train_docs or did in positives[qid] for did in candidates):
                raise ValueError(f"Invalid known-positive BM25 negative for {qid}")
        self.train_queries, self.train_positives, self.train_aliases = query_texts, positives, aliases
        self.train_doc_ids = sorted(train_docs)
        self.train_doc_to_index = {doc_id: index for index, doc_id in enumerate(self.train_doc_ids)}
        self.train_query_ids = sorted(positives)
        self.train_query_to_index = {qid: index for index, qid in enumerate(self.train_query_ids)}
        self.train_positive_sets = {qid: set(docs) for qid, docs in positives.items()}
        self.train_positive_counts = {qid: len(docs) for qid, docs in positives.items()}
        self.dev_queries, self.dev_relevance = dev_queries, dev_relevance
        self.dev_query_ids = sorted(dev_queries)
        self.dev_query_id_to_index = {qid: index for index, qid in enumerate(self.dev_query_ids)}
        self.dev_doc_ids = sorted(dev_docs)
        self.train_dataset = NFCorpusTrainDataset(query_texts, positives, hard_negatives, self.seed)
        self.dev_dataset = NFCorpusDevDataset(dev_queries)
        self.train_collator = NFCorpusCollator(
            self.tokenizer,
            train_docs,
            self.max_question_length,
            self.max_context_length,
            self.context_overlap,
            True,
            self.train_query_to_index,
            self.train_doc_to_index,
        )
        self.dev_collator = NFCorpusCollator(
            self.tokenizer, dev_docs, self.max_question_length, self.max_context_length, self.context_overlap, False
        )

    def dev_document_batches(self, batch_size: int):
        """Yield the full dev corpus in bounded chunk batches once per epoch."""

        if self.dev_dataset is None:
            raise RuntimeError("Call setup before encoding dev documents.")
        rows, indices = [], []
        doc_ids = sorted(self.dev_collator.documents)
        for doc_index, doc_id in enumerate(doc_ids):
            for row in self.dev_collator.chunks(doc_id):
                rows.append(row)
                indices.append(doc_index)
                if len(rows) == batch_size:
                    yield self.dev_collator.tokenize_chunks(rows), indices
                    rows, indices = [], []
        if rows:
            yield self.dev_collator.tokenize_chunks(rows), indices

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Call setup before train_dataloader.")
        # Lightning creates the first iterator before ``on_train_epoch_start``.
        # Initialize the schedule here for resumed/reloaded dataloaders; the
        # module hook updates it before iterators for subsequent epochs.
        if self.trainer is not None:
            self.train_dataset.set_epoch(self.trainer.current_epoch)
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            generator=generator,
            collate_fn=self.train_collator,
        )

    def val_dataloader(self):
        if self.dev_dataset is None:
            raise RuntimeError("Call setup before val_dataloader.")
        return DataLoader(
            self.dev_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.dev_collator,
        )
