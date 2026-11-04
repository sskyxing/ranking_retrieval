"""Fine-tune a local SBERT encoder on NFCorpus train with multi-positive InfoNCE."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import traceback
import uuid

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
import torch.nn.functional as F
from filelock import FileLock
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

try:
    from .modeling import SBERTNLI
    from .retrieval_data import prepare_training_data, build_epoch_batches, contrastive_loss, load_split
    from .chunk_retrieval import ChunkIndex, ChunkRetriever, cached_chunk_backward
    from .train import (atomic_json, append_metric, now, sha256, publish_path, cleanup_child,
                        save_checkpoint, resolve_resume, restore_rng, clean_best_models)
except ImportError:
    from modeling import SBERTNLI
    from retrieval_data import prepare_training_data, build_epoch_batches, contrastive_loss, load_split
    from chunk_retrieval import ChunkIndex, ChunkRetriever, cached_chunk_backward
    from train import (atomic_json, append_metric, now, sha256, publish_path, cleanup_child,
                       save_checkpoint, resolve_resume, restore_rng, clean_best_models)

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from bm25.run_bm25 import BM25, evaluate  # noqa: E402


class EmbeddingRetriever:
    def __init__(self, documents, queries, model, batch_size):
        self.doc_ids = sorted(documents)
        self.documents = F.normalize(model.encode([documents[did] for did in self.doc_ids], batch_size), dim=1)
        texts = list(dict.fromkeys(queries.values()))
        embeddings = F.normalize(model.encode(texts, batch_size), dim=1)
        self.queries = dict(zip(texts, embeddings))

    def search(self, query, k=10):
        scores = self.documents @ self.queries[query]
        ranking = torch.argsort(scores, descending=True, stable=True)[:k]
        return [self.doc_ids[index] for index in ranking.tolist()]


def retrieval_report(model, documents, queries, relevance, batch_size, chunk_overlap=None):
    retriever = (EmbeddingRetriever(documents, queries, model, batch_size) if chunk_overlap is None
                 else ChunkRetriever(documents, queries, model, batch_size, chunk_overlap))
    summary, per_query = evaluate(retriever, queries, relevance)
    summary["document_count"] = len(documents)
    return {"summary": summary, "per_query": per_query}


def export_best(model, output, progress, config, hashes, summary):
    root = output / "best_models"
    root.mkdir(exist_ok=True)
    name = f"step-{progress['global_step']:09d}-{uuid.uuid4().hex[:8]}"
    staging, final = root / (".tmp-" + name), root / name
    complete = False
    try:
        model.export_sentence_transformer(staging, {
            "normalize_embeddings": True,
            "objective": config["loss"],
            "document_encoding": config.get("document_encoding", "single truncated passage"),
            "chunk_overlap": config.get("chunk_overlap"),
            "text_source": config.get("text_source", "provided preprocessed text"),
            "selection": {"split": "nfcorpus-dev", "metric": config.get("selection_metric", "ndcg_at_10"),
                          "score": summary[config.get("selection_metric", "ndcg_at_10")],
                          "global_step": progress["global_step"],
                          "recall_at_10": summary["recall_at_10"], "ndcg_at_10": summary["ndcg_at_10"]},
            "training_provenance": {"config": config, "data_sha256": hashes},
        })
        # A normalization module makes SentenceTransformer's default encode match training/inference.
        modules_path = staging / "modules.json"
        modules = json.loads(modules_path.read_text(encoding="utf-8"))
        modules.append({"idx": 2, "name": "2", "path": "2_Normalize",
                        "type": "sentence_transformers.models.Normalize"})
        (staging / "2_Normalize").mkdir()
        atomic_json(modules_path, modules)
        atomic_json(staging / "dev_metrics.json", summary)
        complete = True
        publish_path(staging, final)
    finally:
        if staging.exists() and not complete:
            cleanup_child(staging, root)
    return final.relative_to(output).as_posix()


def hash_model(directory):
    if not directory.is_dir():
        raise ValueError(f"Local initialization model missing: {directory}")
    if not (directory / "sbert_config.json").is_file():
        raise ValueError("Initialize from a locally trained SBERT export, not an arbitrary model")
    metadata = json.loads((directory / "sbert_config.json").read_text(encoding="utf-8"))
    if metadata.get("pooling") != "mean" or not metadata.get("training_provenance"):
        raise ValueError("Initialization export lacks mean pooling/training provenance")
    return {p.relative_to(directory).as_posix(): sha256(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and ".cache" not in p.parts}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE.parent / "nfcorpus")
    parser.add_argument("--model-dir", type=Path, default=BASE / "models" / "nli-step3000-nfcorpus-init")
    parser.add_argument("--output-dir", type=Path, default=BASE / "runs" / "nfcorpus-retrieval")
    parser.add_argument("--cache-path", type=Path, default=BASE / "data" / "nfcorpus-hard-negatives.json")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-seq-length", type=int, default=130)
    parser.add_argument("--raw-chunks", action="store_true",
                        help="Original query titles/title+abstract; max chunk score; inverse-positive query weighting; independent positive CE")
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--encoder-micro-batch", type=int, default=8,
                        help="Encoder gradient-cache recomputation batch size for raw-chunks mode")
    parser.add_argument("--sampling", choices=("query", "all-positives"), default="query",
                        help="query: one positive per query/epoch; all-positives: cover every eligible positive pair")
    parser.add_argument("--hard-negative-count", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0, help="Smoke-test cap; never evaluate test before all epochs finish")
    parser.add_argument("--resume", nargs="?", const="latest")
    args = parser.parse_args(argv)
    for key in ("threads", "batch_size", "eval_batch_size", "epochs", "max_seq_length",
                "hard_negative_count", "save_steps", "eval_steps"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    if args.max_steps < 0 or args.max_seq_length < 2 or not 0 <= args.warmup_ratio <= 1:
        parser.error("Invalid max-steps, max-seq-length or warmup-ratio")
    if not math.isfinite(args.temperature) or args.temperature <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("temperature and learning-rate must be finite and positive")
    if args.raw_chunks and (args.sampling != "all-positives" or args.encoder_micro_batch < 1
                            or not 0 <= args.chunk_overlap < args.max_seq_length - 2):
        parser.error("raw-chunks requires all-positives, a positive micro-batch and 0 <= overlap < max-length-2")
    return args


def _run(args, stop_requested):
    output = args.output_dir.resolve()
    if not args.resume and ((output / "status.json").exists() or (output / "latest.json").exists()):
        raise FileExistsError("Existing run: use --resume latest or choose a new output directory")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    status = {"pid": os.getpid(), "started_at": now(), "output_dir": str(output),
              "objective": "multi-positive query-to-document InfoNCE"}
    atomic_json(output / "status.json", {**status, "state": "preparing_data", "updated_at": now()})
    data = prepare_training_data(args.data_dir, args.cache_path, args.hard_negative_count, raw_text=args.raw_chunks)
    initialization = hash_model(args.model_dir.resolve())
    epoch_batches = [build_epoch_batches(data, epoch, args.seed, args.batch_size, args.sampling)
                     for epoch in range(args.epochs)]
    batches_by_epoch = [len(batches) for batches in epoch_batches]
    batches_per_epoch = batches_by_epoch[0]
    total_steps = sum(batches_by_epoch)
    eligible_pairs = sum(len(data["train_relevance"][qid]) for qid in data["valid_query_ids"])
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)
    config = {"task": "nfcorpus-retrieval-v1", "model_name": str(args.model_dir.resolve()),
              "initialization_sha256": initialization, "batch_size": args.batch_size,
              "eval_batch_size": args.eval_batch_size, "epochs": args.epochs,
              "learning_rate": args.learning_rate, "temperature": args.temperature,
              "warmup_ratio": args.warmup_ratio, "warmup_steps": warmup_steps,
              "max_seq_length": args.max_seq_length, "threads": args.threads, "seed": args.seed,
              "eval_steps": args.eval_steps, "total_steps": total_steps,
              "train_queries_per_epoch": len(data["valid_query_ids"]),
              "samples_per_query_per_epoch": {"positive": 1, "hard_negative": 1},
              "hard_negative_count": args.hard_negative_count,
              "optimizer": "torch.AdamW bias_correction=True eps=1e-8 weight_decay=0.01",
              "data_manifest": data["manifest"], "max_grad_norm": 1.0,
              "loss": "logsumexp(all candidates) - logsumexp(all known positive candidates)",
              "selection": "dev nDCG@10, linear gains; test only after completion"}
    if args.sampling == "all-positives":
        config.update({"task": "nfcorpus-retrieval-full-positives-v2",
                       "sampling": "all-positives", "schedule_version": 1,
                       "batches_by_epoch": batches_by_epoch,
                       "eligible_positive_pairs": eligible_pairs,
                       "samples_per_query_per_epoch": {"positive": "all known eligible positives exactly once",
                                                        "hard_negative": "one per positive; shuffled pool cycles"},
                       "query_weighting": "positive-pair weighted, not query balanced"})
        config["schedule_sha256"] = hashlib.sha256(
            json.dumps(epoch_batches, separators=(",", ":")).encode()).hexdigest()
    if args.raw_chunks:
        config.update({"task": "nfcorpus-recall-raw-chunks-v3",
                       "text_source": "original query title; original document title + abstract",
                       "document_encoding": "max cosine across all overlapping token chunks; no chunk-count cap",
                       "chunk_overlap": args.chunk_overlap, "encoder_micro_batch": args.encoder_micro_batch,
                       "dropout": "disabled for deterministic two-pass embedding gradient cache",
                       "query_weighting": "N / (Q * number_of_known_positives_for_query); global scale, no batch renormalization",
                       "loss": "per-positive CE; other known positives masked; inverse-positive-count query weights",
                       "selection_metric": "recall_at_10",
                       "selection": "dev Recall@10; ties keep earlier checkpoint; test only after completion"})
        status["objective"] = config["loss"]
    atomic_json(output / "sampling_summary.json", {"sampling": args.sampling,
                "eligible_positive_pairs": eligible_pairs, "batches_by_epoch": batches_by_epoch,
                "examples_by_epoch": [sum(len(batch) for batch in batches) for batches in epoch_batches],
                "short_batches_by_epoch": [sum(len(batch) < args.batch_size for batch in batches) for batches in epoch_batches]})
    hashes = data["manifest"]["source_sha256"]
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(resolve_resume(args.resume, output), weights_only=False, map_location="cpu")
        if checkpoint.get("format_version") != 1 or checkpoint["config"] != config or checkpoint["data_hashes"] != hashes:
            raise ValueError("Resume rejected: configuration, initialization or train/dev data changed")
    atomic_json(output / "run_config.json", {"config": config, "data_sha256": hashes})
    model = SBERTNLI(str(args.model_dir.resolve()), args.max_seq_length).to("cpu")
    if args.max_seq_length > model.encoder.config.max_position_embeddings:
        raise ValueError("max-seq-length exceeds BERT position embeddings")
    # The NLI classification head is not part of this objective.
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(False)
    chunk_index = (ChunkIndex(data["train_documents"], model.tokenizer, args.max_seq_length, args.chunk_overlap)
                   if args.raw_chunks else None)
    if chunk_index is not None:
        chunk_counts = [len(rows) for rows in chunk_index.documents.values()]
        atomic_json(output / "chunking_summary.json", {"documents": len(chunk_counts),
                    "total_chunks": sum(chunk_counts), "max_chunks_per_document": max(chunk_counts),
                    "chunk_length_with_special_tokens": args.max_seq_length, "overlap": args.chunk_overlap,
                    "max_chunk_count_cap": None})
    decay, no_decay = [], []
    for name, parameter in model.encoder.named_parameters():
        if parameter.requires_grad:
            (no_decay if name.endswith("bias") or "LayerNorm.weight" in name else decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    progress = {"global_step": 0, "epoch": 0, "next_batch": 0, "best_model": None,
                "best_dev_ndcg_at_10": None, "best_dev_recall_at_10": None,
                "last_eval_step": -1, "train_seconds": 0.0, "last_loss": None,
                "completed": False, "test_summary": None}
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        progress = checkpoint["progress"]
        if progress["best_model"] and not (output / progress["best_model"]).is_dir():
            raise ValueError("Checkpoint's best model export is missing")
        restore_rng(checkpoint["rng"])
        del checkpoint
    append_metric(output, {"event": "resume" if args.resume else "start",
                           "global_step": progress["global_step"], "config": config})
    status.update({"total_steps": total_steps, "batches_per_epoch": batches_per_epoch, "config": config})
    if args.sampling == "all-positives":
        status.update({"batches_by_epoch": batches_by_epoch, "positive_pairs_per_epoch": eligible_pairs})
    initial_step = progress["global_step"]
    initial_seconds = progress["train_seconds"]
    training_started = time.perf_counter()

    def write_status(state, **extra):
        elapsed = time.perf_counter() - training_started
        steps = progress["global_step"] - initial_step
        seconds_per_step = elapsed / steps if steps else None
        atomic_json(output / "status.json", {**status, **progress, "state": state, "updated_at": now(),
                    "session_seconds": time.perf_counter() - started, "seconds_per_step": seconds_per_step,
                    "estimated_remaining_seconds": seconds_per_step * (total_steps - progress["global_step"]) if seconds_per_step else None,
                    **extra})

    def persist(reason):
        write_status("saving", reason=reason)
        progress["train_seconds"] = initial_seconds + time.perf_counter() - training_started
        saved = save_checkpoint(output, model, optimizer, scheduler, progress, config, hashes)
        clean_best_models(output, progress)
        append_metric(output, {"event": "checkpoint", "reason": reason,
                               "global_step": progress["global_step"], "path": str(saved)})

    def validate():
        write_status("evaluating_dev")
        report = retrieval_report(model, data["dev_documents"], data["dev_queries"],
                                  data["dev_relevance"], args.eval_batch_size,
                                  args.chunk_overlap if args.raw_chunks else None)
        summary = report["summary"]
        progress["last_eval_step"] = progress["global_step"]
        atomic_json(output / f"dev-step-{progress['global_step']:09d}.json", report)
        key = "recall_at_10" if args.raw_chunks else "ndcg_at_10"
        best_key = "best_dev_" + key
        if progress[best_key] is None or summary[key] > progress[best_key]:
            progress["best_model"] = export_best(model, output, progress, config, hashes, summary)
            progress["best_dev_ndcg_at_10"] = summary["ndcg_at_10"]
            progress["best_dev_recall_at_10"] = summary["recall_at_10"]
        append_metric(output, {"event": "nfcorpus_dev", "global_step": progress["global_step"],
                               **summary, "best_ndcg_at_10": progress["best_dev_ndcg_at_10"]})
        print(f"Dev step {progress['global_step']}: Recall@10={summary['recall_at_10']:.6f}, nDCG@10={summary['ndcg_at_10']:.6f}", flush=True)
        write_status("training")

    bar = tqdm(total=total_steps, initial=progress["global_step"], desc="NFCorpus retrieval CPU", unit="step", mininterval=2)
    reason = "completed"
    try:
        model.train()
        if progress["last_eval_step"] < 0:
            bm25_summary, _ = evaluate(BM25(data["dev_documents"]), data["dev_queries"], data["dev_relevance"])
            atomic_json(output / "bm25-dev.json", bm25_summary)
            validate()
            persist("initial_dev_baseline")
        elif args.resume and not progress["completed"]:
            persist("resume_verified")
        while progress["epoch"] < args.epochs:
            batches = epoch_batches[progress["epoch"]]
            batches_per_epoch = len(batches)
            for batch_index in range(progress["next_batch"], batches_per_epoch):
                if stop_requested[0] or (output / "STOP").exists():
                    reason = "keyboard_interrupt" if stop_requested[0] else "stop_file"
                    break
                if args.max_steps and progress["global_step"] >= args.max_steps:
                    reason = "max_steps"
                    break
                batch = batches[batch_index]
                query_ids = [row[0] for row in batch]
                document_ids = list(dict.fromkeys(did for row in batch for did in row[1:]))
                optimizer.zero_grad(set_to_none=True)
                candidate_chunks = None
                if args.raw_chunks:
                    loss_value, candidate_chunks = cached_chunk_backward(
                        model, chunk_index, [data["train_queries"][qid] for qid in query_ids],
                        query_ids, document_ids, [row[1] for row in batch], data["train_relevance"],
                        eligible_pairs, len(data["valid_query_ids"]), args.encoder_micro_batch, args.temperature)
                else:
                    query_features = model.tokenize([data["train_queries"][qid] for qid in query_ids], torch.device("cpu"))
                    document_features = model.tokenize([data["train_documents"][did] for did in document_ids], torch.device("cpu"))
                    loss = contrastive_loss(model.sentence_embeddings(query_features), model.sentence_embeddings(document_features),
                                            query_ids, document_ids, data["train_relevance"], args.temperature)
                    loss.backward()
                    loss_value = float(loss.detach())
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                progress.update({"global_step": progress["global_step"] + 1, "next_batch": batch_index + 1,
                                 "last_loss": loss_value})
                if args.sampling == "all-positives":
                    progress["positive_pairs_seen"] = progress.get("positive_pairs_seen", 0) + len(batch)
                progress["train_seconds"] = initial_seconds + time.perf_counter() - training_started
                append_metric(output, {"event": "train", "global_step": progress["global_step"],
                                       "epoch": progress["epoch"], "batch": batch_index, "loss": progress["last_loss"],
                                       "query_count": len(query_ids), "candidate_count": len(document_ids),
                                       "candidate_chunks": candidate_chunks,
                                       "learning_rate": scheduler.get_last_lr()[0]})
                bar.update(1)
                bar.set_postfix(loss=f"{progress['last_loss']:.4f}")
                write_status("training")
                if progress["global_step"] % args.eval_steps == 0:
                    validate()
                if progress["global_step"] % args.save_steps == 0:
                    persist("periodic")
            if progress["next_batch"] == batches_per_epoch:
                progress["epoch"] += 1
                progress["next_batch"] = 0
                if progress["last_eval_step"] != progress["global_step"]:
                    validate()
                persist("epoch_end")
            if reason != "completed":
                break
        if progress["epoch"] >= args.epochs:
            final = output / "sentence_transformer"
            if not final.exists():
                staging = output / (".tmp-final-" + uuid.uuid4().hex)
                shutil.copytree(output / progress["best_model"], staging)
                publish_path(staging, final)
            if not progress["completed"]:
                write_status("evaluating_test")
                # This is the first point where any test file is opened.
                documents, queries, relevance = load_split(args.data_dir, "test", raw_text=args.raw_chunks)
                selected = SBERTNLI(str(final), args.max_seq_length).to("cpu")
                report = retrieval_report(selected, documents, queries, relevance, args.eval_batch_size,
                                          args.chunk_overlap if args.raw_chunks else None)
                del selected
                report["config"] = {"model_dir": str(final), "max_seq_length": args.max_seq_length,
                                    "selection": config["selection"], "ndcg": "linear gains",
                                    "document_encoding": config.get("document_encoding", "single passage"),
                                    "chunk_overlap": args.chunk_overlap if args.raw_chunks else None,
                                    "text_source": config.get("text_source", "provided preprocessed text"),
                                    "data_sha256": {name: sha256(args.data_dir / name) for name in
                                                     ("test.docs", "test.titles.queries", "test.2-1-0.qrel")}}
                atomic_json(output / "nfcorpus-test-results.json", report)
                bm25_summary, _ = evaluate(BM25(documents), queries, relevance)
                atomic_json(output / "bm25-test.json", bm25_summary)
                progress["test_summary"] = report["summary"]
                progress["completed"] = True
                append_metric(output, {"event": "nfcorpus_test", "global_step": progress["global_step"],
                                       **report["summary"], "bm25": bm25_summary})
            persist("completed")
            write_status("completed", model_dir=str(final))
        else:
            persist(reason)
            write_status("stopped", reason=reason)
    except Exception as error:
        write_status("failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        bar.close()
    print(json.dumps({"state": "completed" if progress["completed"] else "stopped",
                      "global_step": progress["global_step"], "output_dir": str(output)}), flush=True)


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    requested = [False]
    old_handler = signal.getsignal(signal.SIGINT)
    with FileLock(str(output / ".training.lock"), timeout=0):
        signal.signal(signal.SIGINT, lambda *_: requested.__setitem__(0, True))
        try:
            _run(args, requested)
        finally:
            signal.signal(signal.SIGINT, old_handler)


if __name__ == "__main__":
    run(parse_args())
