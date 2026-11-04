"""Offline checks for raw mappings, max-chunk gradients, balanced loss and resume."""
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
try:
    from .modeling import SBERTNLI
    from .chunk_retrieval import (ChunkIndex, ChunkRetriever, features_from_ids,
                                  max_document_scores, per_positive_loss, cached_chunk_backward)
    from .verify_training import tiny_bert, run_cli
    from .verify_retrieval_training import write_split, checkpoint, metrics
    from .retrieval_data import prepare_training_data, load_split
except ImportError:
    from modeling import SBERTNLI
    from chunk_retrieval import (ChunkIndex, ChunkRetriever, features_from_ids,
                                 max_document_scores, per_positive_loss, cached_chunk_backward)
    from verify_training import tiny_bert, run_cli
    from verify_retrieval_training import write_split, checkpoint, metrics
    from retrieval_data import prepare_training_data, load_split


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    with TemporaryDirectory(prefix="sbert-chunk-verification-") as temporary:
        root = Path(temporary)
        pretrained = tiny_bert(root / "pretrained")
        model = SBERTNLI(str(pretrained), 16).eval()
        documents = {"a": "a short sentence " * 14 + "different",
                     "b": "a different sentence", "c": "not the same " * 9}
        index = ChunkIndex(documents, model.tokenizer, 16, 4)
        assert len(index.documents["a"]) > 1
        original = model.tokenizer(documents["a"], add_special_tokens=False)["input_ids"]
        rebuilt = index.documents["a"][0][1:-1]
        for row in index.documents["a"][1:]:
            rebuilt += row[1:-1][4:]
        assert rebuilt == original, "Chunking lost or reordered tail tokens"
        assert all(len(row) <= 16 for chunks in index.documents.values() for row in chunks)
        qids, dids, positives = ["q0", "q1"], ["a", "b", "c"], ["a", "c"]
        relevance = {"q0": {"a": 2, "b": 1}, "q1": {"c": 2}}
        score = torch.tensor([[0.2, 0.9, 0.1], [0.0, 0.1, 0.3]], requires_grad=True)
        loss = per_positive_loss(score, qids, dids, positives, relevance, 3, 2, 1.0)
        first = math.log(math.exp(0.2) + math.exp(0.1)) - 0.2
        second = math.log(math.exp(0) + math.exp(0.1) + math.exp(0.3)) - 0.3
        assert abs(float(loss.detach()) - (0.75 * first + 1.5 * second) / 2) < 1e-6
        loss.backward()
        assert score.grad[0, 1] == 0, "Other known positive incorrectly received negative gradient"
        assert score.grad[0, 0] < 0 and score.grad[0, 2] > 0

        texts = ["a short sentence", "not the same"]
        rows, spans = index.flatten(dids)
        query_rows = model.tokenizer(texts, truncation=True, max_length=16)["input_ids"]
        def direct_encode(token_rows):
            return torch.cat([model.sentence_embeddings(features_from_ids(model.tokenizer, token_rows[i:i + 2]))
                              for i in range(0, len(token_rows), 2)])
        direct_scores = max_document_scores(direct_encode(query_rows), direct_encode(rows), spans)
        direct_loss = per_positive_loss(direct_scores, qids, dids, positives, relevance, 3, 2)
        direct_loss.backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        cached_loss, _ = cached_chunk_backward(model, index, texts, qids, dids, positives, relevance, 3, 2, 2)
        assert abs(cached_loss - float(direct_loss.detach())) < 1e-6
        for name, parameter in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name], rtol=2e-4, atol=2e-6)
        retriever = ChunkRetriever({"doc-z": documents["a"], "doc-a": documents["a"]},
                                   {"q": texts[0]}, model, 2, 4)
        assert retriever.search(texts[0], 10) == ["doc-a", "doc-z"]
        print("PASS: full chunk token coverage, document dedup/ties, per-positive masking, global query weights, cached vs direct encoder gradients", flush=True)

        data = root / "data"
        (data / "raw").mkdir(parents=True)
        write_split(data, "train", {"q0": "filtered short", "q1": "filtered different",
                    "q2": "filtered longer", "q3": "filtered not"},
                    [("q0", "doc-a", 2), ("q0", "doc-b", 1), ("q1", "doc-b", 2),
                     ("q2", "doc-c", 2), ("q3", "doc-d", 2)])
        write_split(data, "dev", {"v0": "filtered same"}, [("v0", "doc-e", 2)])
        raw_docs = {"doc-a": "a short sentence " * 8, "doc-b": "a different sentence",
                    "doc-c": "a longer example " * 7, "doc-d": "not the same",
                    "doc-e": "the same short example"}
        raw_queries = {"q0": "A short sentence!", "q1": "A different sentence?",
                       "q2": "A longer example", "q3": "Not the same",
                       "v0": "The same short example", "t0": "A short sentence"}
        (data / "raw/doc_dump.txt").write_text(
            "".join(f"{did}\turl\tTitle {did}\t{text}\n" for did, text in raw_docs.items()), encoding="utf-8")
        (data / "raw/nfdump.txt").write_text(
            "".join(f"{qid}\turl\t{text}\tignored body\n" for qid, text in raw_queries.items()), encoding="utf-8")
        cache = root / "cache.json"
        prepared = prepare_training_data(data, cache, 2, raw_text=True)
        assert prepared["train_queries"]["q0"] == "A short sentence!"
        assert prepared["train_documents"]["doc-a"].startswith("Title doc-a\n")
        assert "raw/doc_dump.txt" in prepared["manifest"]["source_sha256"]
        initial = root / "initial"
        SBERTNLI(str(pretrained), 16).export_sentence_transformer(initial, {"training_provenance": {"fixture": True}})
        common = ["--data-dir", data, "--model-dir", initial, "--cache-path", cache,
                  "--threads", "1", "--batch-size", "2", "--eval-batch-size", "2",
                  "--epochs", "1", "--max-seq-length", "16", "--hard-negative-count", "2",
                  "--eval-steps", "2", "--save-steps", "2", "--sampling", "all-positives",
                  "--raw-chunks", "--chunk-overlap", "4", "--encoder-micro-batch", "2"]
        continuous, resumed = root / "continuous", root / "resumed"
        run_cli("train_retrieval.py", [*common, "--output-dir", resumed, "--max-steps", "1"])
        assert not any(row["event"] == "nfcorpus_test" for row in metrics(resumed))
        assert not (data / "test.docs").exists()
        write_split(data, "test", {"t0": "filtered test"}, [("t0", "doc-a", 2)])
        run_cli("train_retrieval.py", [*common, "--output-dir", continuous])
        run_cli("train_retrieval.py", [*common, "--output-dir", resumed, "--resume", "latest"])
        left, right = checkpoint(continuous), checkpoint(resumed)
        assert left["config"]["selection_metric"] == "recall_at_10"
        assert left["config"]["task"] == "nfcorpus-recall-raw-chunks-v3"
        assert left["progress"]["positive_pairs_seen"] == right["progress"]["positive_pairs_seen"] == 5
        for name in left["model"]:
            assert torch.equal(left["model"][name], right["model"][name]), f"Resume changed {name}"
        assert torch.equal(left["rng"]["torch"], right["rng"]["torch"])
        assert left["scheduler"] == right["scheduler"]
        assert left["progress"]["test_summary"] == right["progress"]["test_summary"]
        assert [r["loss"] for r in metrics(continuous) if r["event"] == "train"] == [
            r["loss"] for r in metrics(resumed) if r["event"] == "train"]
        metadata = json.loads((continuous / "sentence_transformer/sbert_config.json").read_text())
        assert metadata["selection"]["metric"] == "recall_at_10" and metadata["chunk_overlap"] == 4
        assert sum(row["event"] == "nfcorpus_test" for row in metrics(resumed)) == 1
        print("PASS: original text mappings, raw cache provenance, no early test access, chunk training/export, exact full-state resume and Recall selection", flush=True)
    print("All raw-chunk training checks passed.", flush=True)


if __name__ == "__main__":
    main()
