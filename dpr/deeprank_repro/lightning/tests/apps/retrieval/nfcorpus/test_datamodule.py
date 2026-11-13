import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from smile_lightning.apps.retrieval.nfcorpus.datamodule import (
    NFCorpusCollator,
    NFCorpusDataModule,
    NFCorpusTrainDataset,
    load_train_split,
    merged_training_queries,
)


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.question_texts = None

    @staticmethod
    def _ids(text):
        return [index + 10 for index, _ in enumerate(text.split())]

    def encode(self, text, add_special_tokens=True, verbose=False):
        del verbose
        ids = self._ids(text)
        return [101, *ids, 102] if add_special_tokens else ids

    @staticmethod
    def build_inputs_with_special_tokens(title_ids, abstract_ids):
        return [101, *title_ids, 102, *abstract_ids, 102]

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"
        self.question_texts = list(texts)
        rows = [self.encode(text)[:max_length] for text in texts]
        rows = [row + [0] * (max_length - len(row)) for row in rows]
        input_ids = torch.tensor(rows)
        return {"input_ids": input_ids, "attention_mask": input_ids.ne(0).long()}

    def pad(self, rows, padding, max_length, return_tensors):
        assert padding == "max_length"
        assert return_tensors == "pt"
        input_rows = [row["input_ids"][:max_length] for row in rows]
        input_rows = [row + [0] * (max_length - len(row)) for row in input_rows]
        input_ids = torch.tensor(input_rows)
        return {"input_ids": input_ids, "attention_mask": input_ids.ne(0).long()}


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_data_root(tmp_path):
    root = tmp_path / "nfcorpus"
    _write(root / "train.docs", "d1\tprocessed one\nd2\tprocessed two\nd3\tprocessed three\n")
    _write(root / "train.titles.queries", "q1\tquery one\nq2\tquery two\nq3\tdev duplicate\n")
    _write(
        root / "train.2-1-0.qrel",
        "q1 0 d1 2\nq1 0 d2 1\nq2 0 d2 2\nq3 0 d3 1\n",
    )
    _write(root / "dev.docs", "d1\tprocessed one\nd2\tprocessed two\n")
    _write(root / "dev.titles.queries", "v1\tdev duplicate\nv2\tvalidation query\n")
    _write(root / "dev.2-1-0.qrel", "v1 0 d1 1\nv2 0 d2 2\n")
    _write(
        root / "raw" / "doc_dump.txt",
        "d1\turl1\tTitle One\tAbstract one has several words\n"
        "d2\turl2\tTitle Two\tAbstract two has several words\n"
        "d3\turl3\tTitle Three\tAbstract three has several words\n",
    )
    _write(
        root / "raw" / "nfdump.txt",
        "q1\turl\tQuery One?\textra\n"
        "q2\turl\tQuery Two\textra\n"
        "q3\turl\tDev Duplicate\textra\n"
        "v1\turl\tdev duplicate\textra\n"
        "v2\turl\tValidation Query\textra\n",
    )
    source_names = (
        "train.docs",
        "train.titles.queries",
        "train.2-1-0.qrel",
        "dev.titles.queries",
        "raw/doc_dump.txt",
        "raw/nfdump.txt",
    )
    cache = {
        "manifest": {
            "hard_negative_count": 1,
            "bm25": {"k1": 1.2, "b": 0.75},
            "source_sha256": {name: _sha256(root / name) for name in source_names},
        },
        "hard_negatives": {"q1": ["d3"], "q2": ["d1"]},
    }
    _write(root / "hard_negatives_raw_train.json", json.dumps(cache))
    return root


def test_merged_training_queries_merges_aliases_and_excludes_dev_text():
    queries = {"q1": "Same query", "q2": " same   QUERY ", "q3": "dev query"}
    relevance = {"q1": {"d1": 1}, "q2": {"d1": 2, "d2": 1}, "q3": {"d3": 1}}

    texts, positives, aliases = merged_training_queries(queries, relevance, {"v1": "DEV QUERY"})

    assert texts == {"q1": "Same query"}
    assert positives == {"q1": {"d1": 2, "d2": 1}}
    assert aliases == {"q1": ["q1", "q2"]}


def test_train_dataset_covers_every_positive_and_changes_schedule_by_epoch():
    dataset = NFCorpusTrainDataset(
        {"q1": "one", "q2": "two"},
        {"q1": {"d1": 2, "d2": 1, "d3": 1}, "q2": {"d4": 2, "d5": 1}},
        {"q1": ["n1", "n2"], "q2": ["n3"]},
        seed=7,
    )

    epoch_zero = list(dataset.pairs)
    assert len(epoch_zero) == 5
    assert set(epoch_zero) == {
        ("q1", "d1"),
        ("q1", "d2"),
        ("q1", "d3"),
        ("q2", "d4"),
        ("q2", "d5"),
    }

    dataset.set_epoch(1)

    assert dataset.epoch == 1
    assert set(dataset.pairs) == set(epoch_zero)
    assert dataset.pairs != epoch_zero
    assert dataset[0] == dataset[0]


def test_collator_chunks_documents_and_maps_global_document_indices():
    tokenizer = FakeTokenizer()
    collator = NFCorpusCollator(
        tokenizer,
        {
            "d1": ("short title", "one two three four five six seven"),
            "d2": ("negative", "text"),
        },
        max_question_length=8,
        max_context_length=8,
        overlap=2,
        training=True,
        query_to_index={"q1": 4},
        doc_to_index={"d1": 7, "d2": 9},
    )

    batch = collator([("q1", "Question?", "d1", "d2")])

    assert tokenizer.question_texts == ["Question"]
    assert batch["query_indices"].tolist() == [4]
    assert batch["target_doc_indices"].tolist() == [7]
    assert set(batch["chunk_global_doc_indices"].tolist()) == {7, 9}
    assert batch["context"]["input_ids"].shape[1] == 8
    assert torch.count_nonzero(batch["context"]["token_type_ids"]) == 0


def test_setup_validates_cache_and_dataloader_keeps_all_pairs(tmp_path):
    root = make_data_root(tmp_path)
    module = NFCorpusDataModule(
        str(root),
        batch_size=2,
        val_batch_size=2,
        num_workers=0,
        max_question_length=8,
        max_context_length=8,
        context_overlap=2,
        hard_negative_count=1,
        seed=5,
    )
    module.tokenizer = FakeTokenizer()

    module.setup("fit")
    module.trainer = SimpleNamespace(current_epoch=3)
    loader = module.train_dataloader()

    assert len(module.train_dataset) == 3
    assert module.train_dataset.epoch == 3
    assert loader.drop_last is False
    assert sum(len(batch["query_ids"]) for batch in loader) == 3
    docs, queries, positives, aliases = load_train_split(root)
    assert (len(docs), set(queries), sum(map(len, positives.values())), aliases) == (
        3,
        {"q1", "q2"},
        3,
        {"q1": ["q1"], "q2": ["q2"]},
    )


def test_setup_rejects_stale_hard_negative_cache(tmp_path):
    root = make_data_root(tmp_path)
    cache_path = root / "hard_negatives_raw_train.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    cache["manifest"]["source_sha256"]["train.docs"] = "stale"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    module = NFCorpusDataModule(
        str(root), num_workers=0, max_context_length=8, context_overlap=2, hard_negative_count=1
    )
    module.tokenizer = FakeTokenizer()

    with pytest.raises(ValueError, match="stale or missing source: train.docs"):
        module.setup("fit")


def test_persistent_workers_are_rejected_for_epoch_dependent_schedule(tmp_path):
    with pytest.raises(ValueError, match="persistent_workers=false"):
        NFCorpusDataModule(str(tmp_path), num_workers=1, persistent_workers=True)
