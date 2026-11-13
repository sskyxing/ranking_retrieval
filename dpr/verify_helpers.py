"""Offline verification for downloader selection and streaming metric helpers."""
from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "scripts"))

from download_nq_resources import select_resources  # noqa: E402
from extract_nq_archives import ARCHIVES  # noqa: E402
from verify_precomputed import contains_answer, iter_json_array  # noqa: E402


def main() -> None:
    small = select_resources(["small"])
    assert [item.name for item in small] == ["nq-test", "nq-dev"]
    paper = select_resources(["paper-eval"])
    assert len(paper) == 54
    assert sum(item.expected_bytes for item in paper) == 71_842_579_573
    assert len({item.relative_path for item in paper}) == len(paper)
    assert set(ARCHIVES) == {"wiki", "train", "dev"}

    payload = [{"question": "q1", "ctxs": []}, {"question": "q2", "ctxs": []}]
    stream = StringIO(json.dumps(payload, ensure_ascii=False))
    assert list(iter_json_array(stream, chunk_size=3)) == payload
    assert contains_answer("The capital is São Paulo.", ["SÃO PAULO"])
    assert not contains_answer("A Parisian cafe", ["Paris"])
    print("DPR helper verification passed")


if __name__ == "__main__":
    main()
