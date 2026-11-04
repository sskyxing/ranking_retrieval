"""Offline contract checks with a temporary tiny BERT; no benchmark claims."""

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as functional
from transformers import BertConfig, BertModel, BertTokenizerFast

try:
    from .modeling import SBERTNLI, mean_pool
except ImportError:
    from modeling import SBERTNLI, mean_pool

BASE = Path(__file__).resolve().parent


def verify_checkpoint_io(root):
    try:
        from . import train as training
    except ImportError:
        import train as training

    def windows_error(code):
        error = OSError("simulated Windows filesystem failure")
        error.winerror = code
        return error

    replace = training.os.replace
    for code in (5, 32, 33):
        source, destination = root / f"source-{code}", root / f"published-{code}"
        source.mkdir()
        (source / "marker").write_text("complete", encoding="utf-8")
        failures = iter((windows_error(code), windows_error(code)))

        def flaky_replace(src, dst):
            failure = next(failures, None)
            if failure is not None:
                raise failure
            return replace(src, dst)

        with patch.object(training.os, "replace", side_effect=flaky_replace) as mocked:
            with patch.object(training.time, "sleep"):
                training.publish_path(source, destination)
        assert mocked.call_count == 3
        assert not source.exists() and (destination / "marker").read_text() == "complete"

    for index, error in enumerate((windows_error(5), windows_error(112), ValueError("bad path"))):
        source = root / f"permanent-{index}"
        source.mkdir()
        (source / "marker").write_text("complete", encoding="utf-8")
        with patch.object(training.os, "replace", side_effect=error) as mocked:
            with patch.object(training.time, "sleep"):
                try:
                    training.publish_path(source, root / f"never-published-{index}")
                except type(error) as caught:
                    assert caught is error
                else:
                    raise AssertionError("Permanent publication failure was suppressed")
        assert (2 < mocked.call_count <= 12) if index == 0 else mocked.call_count == 1
        assert (source / "marker").read_text() == "complete"

    output = root / "failed-checkpoint"
    output.mkdir()
    old_pointer = b'{"checkpoint":"checkpoints/previous"}\n'
    (output / "latest.json").write_bytes(old_pointer)
    model = torch.nn.Linear(2, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    progress = {"global_step": 7, "epoch": 0, "next_batch": 7, "best_model": None}
    error = windows_error(5)
    publish = training.publish_path

    def reject_directory(src, dst):
        if src.is_dir():
            raise error
        return publish(src, dst)

    with patch.object(training, "publish_path", side_effect=reject_directory):
        try:
            training.save_checkpoint(output, model, optimizer, scheduler, progress, {}, {})
        except OSError as caught:
            assert caught is error
        else:
            raise AssertionError("Expected directory publication failure")
    retained = list((output / "checkpoints").glob(".tmp-step-*"))
    assert len(retained) == 1, "Complete unpublished checkpoint was discarded"
    assert json.loads((retained[0] / "summary.json").read_text())["global_step"] == 7
    state = torch.load(retained[0] / "state.pt", weights_only=False, map_location="cpu")
    assert state["progress"] == progress
    assert {"model", "optimizer", "scheduler", "rng"}.issubset(state)
    assert (output / "latest.json").read_bytes() == old_pointer
    assert not list((output / "checkpoints").glob("step-*"))

    incomplete = root / "incomplete-checkpoint"
    incomplete.mkdir()
    with patch.object(training.torch, "save", side_effect=RuntimeError("simulated incomplete write")):
        try:
            training.save_checkpoint(incomplete, model, optimizer, scheduler, progress, {}, {})
        except RuntimeError:
            pass
        else:
            raise AssertionError("Expected serialization failure")
    assert not list((incomplete / "checkpoints").iterdir())
    print("PASS: transient Windows locks retry; permanent failures stay bounded; complete failed checkpoint preserved; incomplete write cleaned", flush=True)


def tiny_bert(directory):
    directory.mkdir()
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "short",
                  "sentence", "longer", "example", "different", "not", "the", "same"]
    vocab_file = directory / "vocab.txt"
    vocab_file.write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True)
    tokenizer.save_pretrained(directory)
    model = BertModel(BertConfig(
        vocab_size=len(vocabulary), hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=24, max_position_embeddings=32,
    ))
    model.save_pretrained(directory)
    return directory


def run_cli(script, arguments, succeeds=True):
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(BASE / script), *map(str, arguments)],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    if (result.returncode == 0) != succeeds:
        raise AssertionError(f"Unexpected exit code {result.returncode}\n{result.stdout}\n{result.stderr}")
    return result


def verify_encoder_and_retrieval(root, pretrained):
    torch.manual_seed(42)
    model = SBERTNLI(str(pretrained), max_seq_length=12)
    first = model.tokenize(["a short sentence", "a longer sentence example"], torch.device("cpu"))
    second = model.tokenize(["a different sentence", "not the same"], torch.device("cpu"))
    encoder_before = model.encoder.embeddings.word_embeddings.weight.detach().clone()
    classifier_before = model.classifier.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    functional.cross_entropy(model(first, second), torch.tensor([0, 2])).backward()
    assert model.encoder.embeddings.word_embeddings.weight.grad is not None, "Missing encoder gradient"
    assert model.classifier.weight.grad is not None, "Missing classifier gradient"
    optimizer.step()
    assert not torch.equal(encoder_before, model.encoder.embeddings.word_embeddings.weight), "Encoder did not update"
    assert not torch.equal(classifier_before, model.classifier.weight), "Classifier did not update"

    # Padding must be ignored, while every non-padding token participates.
    tokens = torch.tensor([[[2.0, 4.0], [4.0, 8.0], [999.0, 999.0]]])
    assert torch.equal(mean_pool(tokens, torch.tensor([[1, 1, 0]])), torch.tensor([[3.0, 6.0]]))
    sentences = ["a short sentence", "a longer sentence example", "not the same"]
    single = model.encode(sentences, batch_size=1)
    batched = model.encode(sentences, batch_size=3)
    torch.testing.assert_close(single, batched, rtol=1e-5, atol=1e-6)
    exported = root / "trained-export"
    model.export_sentence_transformer(exported, {
        "training_provenance": {"fixture": "tiny-random-BERT; one SGD update; no benchmark claim"},
    })
    reloaded = SBERTNLI(str(exported), max_seq_length=12)
    torch.testing.assert_close(single, reloaded.encode(sentences, batch_size=2), rtol=1e-5, atol=1e-6)

    data = root / "retrieval-data"
    data.mkdir()
    (data / "test.docs").write_text(
        "doc-z\ta short sentence\ndoc-a\ta short sentence\ndoc-b\ta short sentence\n", encoding="utf-8"
    )
    (data / "test.titles.queries").write_text(
        "q1\ta short sentence\nq-empty\ta short sentence\n", encoding="utf-8"
    )
    (data / "test.2-1-0.qrel").write_text("q1 0 doc-z 2\nq-empty 0 doc-a 0\n", encoding="utf-8")
    report_file = root / "retrieval-results.json"
    arguments = ["--data-dir", data, "--output", report_file, "--threads", "1", "--batch-size", "2"]
    run_cli("evaluate_retrieval.py", ["--model-dir", exported, *arguments])
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["per_query"]["q1"]["top_10_doc_ids"] == ["doc-a", "doc-b", "doc-z"]
    assert report["summary"]["recall_at_10"] == 1.0
    assert report["summary"]["ndcg_at_10"] == 0.5
    assert report["summary"]["skipped_query_ids"] == ["q-empty"]
    assert report["config"]["max_seq_length"] == 12
    assert report["config"]["model_dir"] == str(exported.resolve())
    scores = report["per_query"]["q1"]["top_10_scores"]
    assert len(scores) == 3 and scores[0] == scores[1] == scores[2]

    absent_output = root / "must-not-exist.json"
    for rejected_model in (root / "missing-model", pretrained):
        result = run_cli("evaluate_retrieval.py", [
            "--model-dir", rejected_model, "--output", absent_output, "--data-dir", data,
        ], succeeds=False)
        assert "ValueError" in result.stderr
        assert not absent_output.exists(), "Evaluation created a result without training provenance"
    print("PASS: encoder/classifier update, padding invariance, export round-trip, exact ties, shared metrics, and missing-export rejection", flush=True)


def verify_training_resume(root, pretrained):
    data = root / "training-data"
    data.mkdir()
    nli = [dict(
        split="train", label=["entailment", "neutral", "contradiction"][index % 3],
        sentence1="a " + ["short sentence", "longer sentence example", "different sentence"][index % 3],
        sentence2=["a short sentence", "not the same", "a different example", "the same"][index % 4],
    ) for index in range(12)]
    stsb = [dict(
        split=split, score=str(index),
        sentence1=["a short sentence", "a different example", "a longer sentence", "not the same"][index],
        sentence2=["a short sentence", "a short sentence", "a short example", "the same"][index],
    ) for split in ("dev", "test") for index in range(4)]
    for filename, records in (("AllNLI.tsv.gz", nli), ("stsbenchmark.tsv.gz", stsb)):
        with gzip.open(data / filename, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), delimiter="\t", quoting=csv.QUOTE_NONE)
            writer.writeheader()
            writer.writerows(records)
    common = [
        "--model-name", pretrained, "--data-dir", data, "--batch-size", "2",
        "--eval-batch-size", "2", "--max-seq-length", "16", "--threads", "1",
        "--eval-steps", "3", "--save-steps", "2",
    ]
    continuous = root / "continuous-training"
    resumed = root / "resumed-training"
    run_cli("train.py", [*common, "--output-dir", continuous])
    run_cli("train.py", [*common, "--output-dir", resumed, "--max-steps", "2"])
    stopped = json.loads((resumed / "status.json").read_text(encoding="utf-8"))
    assert stopped["state"] == "stopped" and stopped["global_step"] == 2
    interrupted_metrics = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert not any(row["event"] == "stsb_test" for row in interrupted_metrics), "Interrupted run evaluated test data"
    run_cli("train.py", [*common, "--output-dir", resumed, "--resume", "latest"])

    def load_checkpoint(directory):
        pointer = json.loads((directory / "latest.json").read_text(encoding="utf-8"))
        return torch.load(directory / pointer["checkpoint"] / "state.pt", weights_only=False, map_location="cpu")

    first, second = load_checkpoint(continuous), load_checkpoint(resumed)
    assert first["model"].keys() == second["model"].keys()
    for key in first["model"]:
        assert torch.equal(first["model"][key], second["model"][key]), f"Resume changed model tensor {key}"
    assert torch.equal(first["rng"]["torch"], second["rng"]["torch"]), "Resume changed Torch RNG"
    assert first["scheduler"] == second["scheduler"], "Resume changed scheduler state"
    assert first["progress"]["test_spearman"] == second["progress"]["test_spearman"]
    assert first["progress"]["global_step"] == second["progress"]["global_step"] == 6
    first_metrics, second_metrics = [
        [json.loads(line) for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        for directory in (continuous, resumed)
    ]
    assert [row["loss"] for row in first_metrics if row["event"] == "train"] == [
        row["loss"] for row in second_metrics if row["event"] == "train"
    ], "Resume changed per-step training losses"
    for directory, metrics in ((continuous, first_metrics), (resumed, second_metrics)):
        assert sum(row["event"] == "stsb_test" for row in metrics) == 1, "Final test must run once"
        status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        assert status["state"] == "completed"
        assert (directory / "sentence_transformer" / "sbert_config.json").is_file()
        checkpoints = list((directory / "checkpoints").glob("step-*"))
        assert len(checkpoints) <= 2, "Checkpoint retention exceeded two snapshots"
    print("PASS: continuous versus interrupted/resumed model tensors, per-step losses, scheduler, Torch RNG and final test score are identical", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-training", action="store_true",
                        help="Also compare tiny continuous and resumed training runs (6 updates each)")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    with TemporaryDirectory(prefix="sbert-verification-") as temporary:
        root = Path(temporary)
        verify_checkpoint_io(root)
        pretrained = tiny_bert(root / "pretrained")
        verify_encoder_and_retrieval(root, pretrained)
        if args.include_training:
            verify_training_resume(root, pretrained)
    print("All checks passed; temporary fixtures removed. No real training run or benchmark result was modified.", flush=True)


if __name__ == "__main__":
    main()
