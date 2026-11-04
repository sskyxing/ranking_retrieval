"""Offline retrieval training, leakage, gradient and exact-resume regression checks."""
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
try:
    from .verify_training import tiny_bert, run_cli
    from .modeling import SBERTNLI
    from .retrieval_data import prepare_training_data, build_epoch_examples, build_epoch_batches, contrastive_loss
except ImportError:
    from verify_training import tiny_bert, run_cli
    from modeling import SBERTNLI
    from retrieval_data import prepare_training_data, build_epoch_examples, build_epoch_batches, contrastive_loss


def write_split(data, split, queries, judgments):
    documents = {"doc-a": "a short sentence", "doc-b": "a different sentence",
                 "doc-c": "a longer example", "doc-d": "not the same",
                 "doc-e": "the same short example"}
    for suffix, rows in (("docs", documents), ("titles.queries", queries)):
        (data / f"{split}.{suffix}").write_text(
            "".join(f"{key}\t{value}\n" for key, value in rows.items()), encoding="utf-8")
    (data / f"{split}.2-1-0.qrel").write_text(
        "".join(f"{qid} 0 {did} {grade}\n" for qid, did, grade in judgments), encoding="utf-8")


def checkpoint(output):
    pointer = json.loads((output / "latest.json").read_text())
    return torch.load(output / pointer["checkpoint"] / "state.pt", weights_only=False, map_location="cpu")


def metrics(output):
    return [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    with TemporaryDirectory(prefix="sbert-retrieval-verification-") as temporary:
        root = Path(temporary)
        data = root / "data"
        data.mkdir()
        write_split(data, "train", {"q0": "a short sentence", "q1": "a different sentence",
                    "q2": "a longer example", "q3": "not same",
                    "qdup": "A  short sentence", "qoverlap": "THE  SAME"},
                    [("q0", "doc-a", 2), ("qdup", "doc-b", 1),
                     ("q1", "doc-b", 2), ("q2", "doc-c", 2),
                     ("q3", "doc-d", 2), ("qoverlap", "doc-e", 1)])
        write_split(data, "dev", {"v0": "the same", "v1": "a short example"},
                    [("v0", "doc-e", 2), ("v1", "doc-a", 2), ("v1", "doc-b", 1)])
        cache = root / "negatives.json"
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            assert not path.name.startswith("test."), "Preparation read test data"
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            prepared = prepare_training_data(data, cache, 2)
        assert prepared["valid_query_ids"] == ["q0", "q1", "q2", "q3"]
        assert prepared["train_relevance"]["q0"] == {"doc-a": 2, "doc-b": 1}
        assert prepared["manifest"]["excluded_dev_overlap_query_ids"] == ["qoverlap"]
        for qid, negatives in prepared["hard_negatives"].items():
            assert not set(negatives).intersection(prepared["train_relevance"][qid])
        assert build_epoch_examples(prepared, 0, 42) == build_epoch_examples(prepared, 0, 42)
        assert build_epoch_examples(prepared, 0, 42) != build_epoch_examples(prepared, 1, 42)
        expected_pairs = {(qid, did) for qid in prepared["valid_query_ids"]
                          for did in prepared["train_relevance"][qid]}
        full_batches = build_epoch_batches(prepared, 0, 42, 2, "all-positives")
        actual_pairs = [(qid, did) for batch in full_batches for qid, did, _ in batch]
        assert len(actual_pairs) == len(set(actual_pairs)) == len(expected_pairs)
        assert set(actual_pairs) == expected_pairs
        assert all(len(batch) == len({row[0] for row in batch}) for batch in full_batches)
        assert full_batches == build_epoch_batches(prepared, 0, 42, 2, "all-positives")
        assert full_batches != build_epoch_batches(prepared, 1, 42, 2, "all-positives")
        assert any(len(batch) == 1 for batch in full_batches), "Short explicit-negative batch not tested"
        assert all(negative not in prepared["train_relevance"][qid]
                   for batch in full_batches for qid, _, negative in batch)
        skewed = dict(prepared)
        skewed["valid_query_ids"] = ["q0", "q1"]
        skewed["train_relevance"] = {"q0": {f"long-{i}": 1 for i in range(7)}, "q1": {"only": 1}}
        skewed["hard_negatives"] = {"q0": ["neg-a"], "q1": ["neg-b"]}
        skewed_batches = build_epoch_batches(skewed, 0, 42, 2, "all-positives")
        assert len(skewed_batches) == 7 and sum(map(len, skewed_batches)) == 8
        assert all(len(batch) == len({row[0] for row in batch}) for batch in skewed_batches)
        reloaded = prepare_training_data(data, cache, 2)
        assert reloaded["manifest"] == prepared["manifest"]
        old_hash = prepared["manifest"]["data_hash"]
        with (data / "train.docs").open("a") as stream:
            stream.write("doc-extra\ta different example\n")
        assert prepare_training_data(data, cache, 2)["manifest"]["data_hash"] != old_hash

        q = torch.tensor([[1.0, 0.0]], requires_grad=True)
        docs = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        relevance = {"q": {"a": 2, "b": 1}}
        loss = contrastive_loss(q, docs, ["q"], ["a", "b", "c"], relevance, 1.0)
        expected = math.log(2 * math.e + 1) - math.log(2 * math.e)
        assert abs(float(loss.detach()) - expected) < 1e-6, "Known second positive was treated as negative"
        loss.backward()
        assert torch.isfinite(q.grad).all() and torch.isfinite(docs.grad).all()
        worse = contrastive_loss(torch.tensor([[0.0, 1.0]]), docs.detach(), ["q"],
                                  ["a", "b", "c"], relevance, 1.0)
        assert loss < worse
        try:
            contrastive_loss(q, docs, ["q"], ["a", "a", "c"], relevance)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate document IDs accepted")
        print("PASS: train/dev isolation, overlap removal, duplicate-query positive merge, hard-negative exclusion, cache invalidation, multi-positive loss/gradients", flush=True)

        pretrained = tiny_bert(root / "pretrained")
        initial = root / "initial"
        SBERTNLI(str(pretrained), 16).export_sentence_transformer(
            initial, {"training_provenance": {"fixture": "tiny random BERT"}})
        common = ["--data-dir", data, "--model-dir", initial, "--cache-path", cache,
                  "--threads", "1", "--batch-size", "2", "--eval-batch-size", "2",
                  "--epochs", "2", "--max-seq-length", "16", "--hard-negative-count", "2",
                  "--eval-steps", "2", "--save-steps", "2"]
        resumed, continuous = root / "resumed", root / "continuous"
        # Test files do not exist yet: capped training must finish without touching them.
        run_cli("train_retrieval.py", [*common, "--output-dir", resumed, "--max-steps", "1"])
        stopped = json.loads((resumed / "status.json").read_text())
        assert stopped["state"] == "stopped" and stopped["global_step"] == 1
        assert not any(row["event"] == "nfcorpus_test" for row in metrics(resumed))
        write_split(data, "test", {"t0": "a short sentence", "t1": "not the same"},
                    [("t0", "doc-a", 2), ("t1", "doc-d", 2)])
        run_cli("train_retrieval.py", [*common, "--output-dir", continuous])
        run_cli("train_retrieval.py", [*common, "--output-dir", resumed, "--resume", "latest"])
        first, second = checkpoint(continuous), checkpoint(resumed)
        assert first["progress"]["global_step"] == second["progress"]["global_step"] == 4
        for key in first["model"]:
            assert torch.equal(first["model"][key], second["model"][key]), f"Resume changed {key}"
        assert first["scheduler"] == second["scheduler"]
        assert torch.equal(first["rng"]["torch"], second["rng"]["torch"])
        assert first["progress"]["test_summary"] == second["progress"]["test_summary"]
        for output in (continuous, resumed):
            events = metrics(output)
            assert sum(row["event"] == "nfcorpus_test" for row in events) == 1
            assert json.loads((output / "status.json").read_text())["state"] == "completed"
            assert json.loads((output / "sentence_transformer" / "sbert_config.json").read_text())["selection"]["split"] == "nfcorpus-dev"
            assert len(list((output / "checkpoints").glob("step-*"))) <= 2
        before = SBERTNLI(str(initial), 16)
        assert not torch.equal(before.encoder.embeddings.word_embeddings.weight,
                               first["model"]["encoder.embeddings.word_embeddings.weight"])
        assert [r["loss"] for r in metrics(continuous) if r["event"] == "train"] == [
            r["loss"] for r in metrics(resumed) if r["event"] == "train"]
        run_cli("train_retrieval.py", [*common, "--output-dir", resumed, "--resume", "latest"])
        assert sum(row["event"] == "nfcorpus_test" for row in metrics(resumed)) == 1
        print("PASS: encoder updates, initial/dev evaluation, full resume tensor/loss/RNG equality, final dev selection, test once, checkpoint retention", flush=True)

        # Full-positive mode changes the number/contents of batches. Verify its
        # schedule, cursor and RNG produce the same weights across a restart.
        full_common = [*common, "--sampling", "all-positives", "--max-seq-length", "24"]
        full_resumed, full_continuous = root / "full-resumed", root / "full-continuous"
        run_cli("train_retrieval.py", [*full_common, "--output-dir", full_continuous])
        run_cli("train_retrieval.py", [*full_common, "--output-dir", full_resumed, "--max-steps", "1"])
        run_cli("train_retrieval.py", [*full_common, "--output-dir", full_resumed, "--resume", "latest"])
        left, right = checkpoint(full_continuous), checkpoint(full_resumed)
        for key in left["model"]:
            assert torch.equal(left["model"][key], right["model"][key]), f"Full-positive resume changed {key}"
        assert left["scheduler"] == right["scheduler"]
        assert torch.equal(left["rng"]["torch"], right["rng"]["torch"])
        assert left["progress"]["positive_pairs_seen"] == right["progress"]["positive_pairs_seen"] == 10
        assert left["progress"]["test_summary"] == right["progress"]["test_summary"]
        assert left["config"]["sampling"] == "all-positives"
        assert left["config"]["max_seq_length"] == 24
        assert left["config"]["total_steps"] == 6
        assert left["config"]["schedule_sha256"] == right["config"]["schedule_sha256"]
        assert [row["loss"] for row in metrics(full_continuous) if row["event"] == "train"] == [
            row["loss"] for row in metrics(full_resumed) if row["event"] == "train"]
        final_metadata = json.loads((full_continuous / "sentence_transformer" / "sbert_config.json").read_text())
        assert final_metadata["max_seq_length"] == 24
        assert sum(row["event"] == "nfcorpus_test" for row in metrics(full_resumed)) == 1
        long_model = SBERTNLI(str(full_continuous / "sentence_transformer"), 24)
        assert long_model.tokenize(["short " * 50], torch.device("cpu"))["input_ids"].shape[1] == 24
        print("PASS: full-positive exact coverage, unique queries per batch, skewed/short batches, longer tokenization/export, deterministic full-positive resume", flush=True)
    print("All retrieval checks passed; no real training or benchmark data modified.", flush=True)


if __name__ == "__main__":
    main()
