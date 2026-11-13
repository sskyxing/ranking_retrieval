import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[3] / "scripts" / "retrieval" / "diagnose_dpr_nfcorpus_dev.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("diagnose_dpr_nfcorpus_dev_under_test", SCRIPT)
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


def test_dot_and_cosine_use_same_chunks_but_can_change_document_ranking():
    questions = torch.tensor([[1.0, 0.0]])
    passages = torch.tensor([[4.0, 3.0], [0.5, 0.2], [3.0, 0.0]])
    scores, question_norms, passage_norms = DIAGNOSTIC.compare_scoring(
        questions, passages, [0, 0, 1], 2
    )

    assert scores["dot"].tolist() == [[4.0, 3.0]]
    assert scores["cosine"].argmax(dim=1).tolist() == [1]
    assert question_norms.tolist() == [1.0]
    assert passage_norms.shape == (3,)
    summary = DIAGNOSTIC.summarize_bias(scores, question_norms, passage_norms, [0, 0, 1], 2)
    assert summary["document_chunk_count"]["count"] == 2
    assert summary["document_chunk_count"]["max"] == 2
    assert summary["mean_dot_cosine_top_10_overlap"] == 2


def test_cosine_rejects_zero_norm_and_missing_passage_document_index():
    with pytest.raises(ValueError, match="nonzero"):
        DIAGNOSTIC.compare_scoring(torch.tensor([[0.0, 0.0]]), torch.tensor([[1.0, 0.0]]), [0], 1)
    with pytest.raises(ValueError, match="Every passage"):
        DIAGNOSTIC.compare_scoring(torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]), [], 1)


def test_parse_args_is_dev_only_and_requires_valid_chunk_configuration(tmp_path):
    base = [
        "--checkpoint", str(tmp_path / "best.ckpt"),
        "--data-dir", str(tmp_path),
        "--output", str(tmp_path / "out.json"),
    ]
    args = DIAGNOSTIC.parse_args(base)
    assert (args.max_length, args.overlap) == (256, 64)
    with pytest.raises(SystemExit):
        DIAGNOSTIC.parse_args(base + ["--max-length", "8", "--overlap", "5"])
