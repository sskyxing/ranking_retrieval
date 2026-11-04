"""Evaluate a raw-text max-chunk model at document level, using its saved chunk settings."""
import argparse
import json
from pathlib import Path
import torch

try:
    from .modeling import SBERTNLI
    from .retrieval_data import load_split
    from .train_retrieval import retrieval_report
    from .train import atomic_json
except ImportError:
    from modeling import SBERTNLI
    from retrieval_data import load_split
    from train_retrieval import retrieval_report
    from train import atomic_json

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=BASE / "runs/nfcorpus-recall-raw-chunks/sentence_transformer")
    parser.add_argument("--data-dir", type=Path, default=BASE.parent / "nfcorpus")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    metadata = json.loads((args.model_dir / "sbert_config.json").read_text(encoding="utf-8"))
    if metadata.get("chunk_overlap") is None or not metadata.get("training_provenance"):
        parser.error("A trained max-chunk export with provenance is required")
    if args.threads < 1 or args.batch_size < 1:
        parser.error("threads/batch-size must be positive")
    torch.set_num_threads(args.threads)
    documents, queries, relevance = load_split(args.data_dir, args.split, raw_text=True)
    model = SBERTNLI(str(args.model_dir), metadata["max_seq_length"]).eval()
    report = retrieval_report(model, documents, queries, relevance, args.batch_size, metadata["chunk_overlap"])
    report["config"] = {"split": args.split, "model_dir": str(args.model_dir.resolve()),
                        "training_export": metadata, "aggregation": "maximum chunk cosine, then document top10"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
