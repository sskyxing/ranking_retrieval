import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[3] / "scripts" / "retrieval" / "evaluate_dpr_nfcorpus.py"
SPEC = importlib.util.spec_from_file_location("evaluate_dpr_nfcorpus_under_test", SCRIPT)
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def test_training_provenance_is_explicit_and_preserved(tmp_path):
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name in ("test.docs", "test.titles.queries", "test.2-1-0.qrel"):
        (data_dir / name).write_text(name, encoding="utf-8")

    args = EVALUATOR.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--output",
            str(tmp_path / "report.json"),
            "--training-data",
            "  NQ warm start; NFCorpus train fine-tuning  ",
            "--source-aml-run-id",
            "fine-tune-run",
            "--text-mode",
            "preprocessed-single",
        ]
    )
    config = EVALUATOR.build_report_config(
        args,
        epoch=4,
        global_step=123,
        data_names=["test.docs", "test.titles.queries", "test.2-1-0.qrel"],
        passage_stats={"passage_count": 3},
    )

    assert config["training_data"] == "NQ warm start; NFCorpus train fine-tuning"
    assert config["source_aml_run_id"] == "fine-tune-run"
    assert config["checkpoint_epoch"] == 4
    assert config["checkpoint_global_step"] == 123
    assert config["checkpoint_bytes"] == len(b"checkpoint")
    assert config["similarity_mode"] == "dot"
    assert config["overlap"] is None
    assert set(config["data_sha256"]) == {"test.docs", "test.titles.queries", "test.2-1-0.qrel"}


def test_training_data_is_required(tmp_path):
    with pytest.raises(SystemExit):
        EVALUATOR.parse_args(
            [
                "--checkpoint",
                str(tmp_path / "best.ckpt"),
                "--data-dir",
                str(tmp_path),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )


def test_scoring_follows_checkpoint_similarity_mode():
    questions = torch.tensor([[1.0, 0.0]])
    passages = torch.tensor([[4.0, 3.0], [3.0, 0.0]])
    dot = EVALUATOR.score_passages(questions, passages, "dot")
    cosine = EVALUATOR.score_passages(questions, passages, "cosine")

    assert dot.argmax(dim=1).tolist() == [0]
    assert cosine.argmax(dim=1).tolist() == [1]
    assert cosine.max() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="similarity mode"):
        EVALUATOR.score_passages(questions, passages, "unknown")


@pytest.mark.parametrize(
    "hyper_parameters,expected",
    [({}, "dot"), ({"similarity_mode": "cosine"}, "cosine")],
)
def test_load_encoders_reads_similarity_mode_and_keeps_old_result_shape(
    tmp_path, monkeypatch, hyper_parameters, expected
):
    class FakeBert:
        def __init__(self, _config):
            self.weights = None

        def load_state_dict(self, weights, strict):
            assert strict
            self.weights = weights

        def eval(self):
            return self

    checkpoint = {
        "epoch": 2,
        "global_step": 5172,
        "hyper_parameters": hyper_parameters,
        "state_dict": {
            "model.question_encoder.weight": torch.tensor([1.0]),
            "model.context_encoder.weight": torch.tensor([2.0]),
        },
    }
    monkeypatch.setattr(EVALUATOR.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(EVALUATOR, "BertModel", FakeBert)
    monkeypatch.setattr(EVALUATOR, "BertConfig", lambda: None)

    old_shape = EVALUATOR.load_encoders(tmp_path / "best.ckpt")
    extended = EVALUATOR.load_encoders(tmp_path / "best.ckpt", return_similarity_mode=True)
    assert len(old_shape) == 4
    assert len(extended) == 5
    assert extended[-1] == expected
    assert extended[0].weights["weight"].item() == 1.0
    assert extended[1].weights["weight"].item() == 2.0
