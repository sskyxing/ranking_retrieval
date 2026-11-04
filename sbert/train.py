"""CPU reproduction of Sentence-BERT's AllNLI softmax training recipe.

Checkpoints contain the whole training state and the next batch cursor. Use
--resume latest to continue after stopping this process or creating output/STOP.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import shutil
import signal
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from filelock import FileLock
from torch.optim import Optimizer
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

try:
    from .modeling import SBERTNLI
except ImportError:
    from modeling import SBERTNLI

BASE = Path(__file__).resolve().parent
LABELS = {"contradiction": 0, "entailment": 1, "neutral": 2}


class OriginalAdamW(Optimizer):
    """The original Transformers AdamW update with correct_bias=False.

The recipe predates torch.optim.AdamW; PyTorch's optimizer always corrects
the moments' bias. Weight decay is applied after the adaptive update.
"""
    def __init__(self, params, lr: float = 2e-5, eps: float = 1e-6) -> None:
        super().__init__(params, dict(lr=lr, betas=(0.9, 0.999), eps=eps, weight_decay=0.01))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("OriginalAdamW does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                average, square_average = state["exp_avg"], state["exp_avg_sq"]
                average.mul_(beta1).add_(gradient, alpha=1 - beta1)
                square_average.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                denominator = square_average.sqrt().add_(group["eps"])
                parameter.addcdiv_(average, denominator, value=-group["lr"])
                if group["weight_decay"]:
                    parameter.add_(parameter, alpha=-group["lr"] * group["weight_decay"])
        return loss


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def retry_filesystem(operation):
    """Retry transient Windows file locks without weakening atomic publication."""
    for attempt in range(10):
        try:
            return operation()
        except OSError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 9:
                raise
            time.sleep(min(0.1 * 2 ** attempt, 2.0))


def publish_path(source: Path, destination: Path) -> None:
    retry_filesystem(lambda: os.replace(source, destination))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        publish_path(temporary, path)
    finally:
        if temporary.exists():
            try:
                retry_filesystem(lambda: temporary.unlink(missing_ok=True))
            except OSError as error:
                print(f"Could not clean temporary JSON {temporary}: {error}", file=sys.stderr, flush=True)


def append_metric(directory: Path, value: dict[str, Any]) -> None:
    with (directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"timestamp": now(), **value}, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_data_file(directory: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        if (directory / name).is_file():
            return directory / name
    raise FileNotFoundError(f"Missing data file in {directory}; expected one of {names}")


def rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)


def read_nli(path: Path) -> list[tuple[str, str, int]]:
    result = []
    for row in rows(path):
        if row["split"] == "train":
            label = row["label"]
            if label not in LABELS:
                raise ValueError(f"Unknown NLI label: {label!r}")
            result.append((row["sentence1"], row["sentence2"], LABELS[label]))
    if not result:
        raise ValueError("NLI training split is empty")
    return result


def read_stsb(path: Path) -> dict[str, list[tuple[str, str, float]]]:
    result = {"dev": [], "test": []}
    for row in rows(path):
        split = row["split"]
        if split in result:
            result[split].append((row["sentence1"], row["sentence2"], float(row["score"])))
    if not result["dev"] or not result["test"]:
        raise ValueError("STS Benchmark dev and test splits must both be present")
    return result


@torch.inference_mode()
def evaluate_stsb(model: SBERTNLI, examples: list[tuple[str, str, float]], batch_size: int) -> float:
    sentences = list(dict.fromkeys(text for row in examples for text in row[:2]))
    embeddings = F.normalize(model.encode(sentences, batch_size=batch_size), p=2, dim=1)
    index = {sentence: offset for offset, sentence in enumerate(sentences)}
    first = torch.tensor([index[row[0]] for row in examples])
    second = torch.tensor([index[row[1]] for row in examples])
    similarities = (embeddings[first] * embeddings[second]).sum(1).numpy()
    correlation = float(spearmanr([row[2] for row in examples], similarities).statistic)
    if not math.isfinite(correlation):
        raise RuntimeError("STS Benchmark Spearman is not finite")
    return correlation


def rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def remove_child(directory: Path, root: Path) -> None:
    if directory.resolve().parent != root.resolve():
        raise ValueError(f"Refusing to remove a directory outside {root}")
    retry_filesystem(lambda: shutil.rmtree(directory))


def cleanup_child(directory: Path, root: Path) -> None:
    """Retention is best-effort; a locked old snapshot must not stop training."""
    try:
        remove_child(directory, root)
    except OSError as error:
        print(f"Retaining {directory}: cleanup failed: {error}", file=sys.stderr, flush=True)


def save_checkpoint(
    output: Path, model: SBERTNLI, optimizer: Optimizer, scheduler,
    progress: dict[str, Any], config: dict[str, Any], hashes: dict[str, str],
) -> Path:
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    name = f"step-{progress['global_step']:09d}-{uuid.uuid4().hex[:8]}"
    staging, final = checkpoints / (".tmp-" + name), checkpoints / name
    staging.mkdir()
    state = {
        "format_version": 1, "config": config, "data_hashes": hashes,
        "progress": progress, "rng": rng_state(),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "saved_at": now(),
    }
    complete_staging = False
    try:
        with (staging / "state.pt").open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(staging / "summary.json", {
            "global_step": progress["global_step"], "epoch": progress["epoch"],
            "next_batch": progress["next_batch"], "saved_at": state["saved_at"],
            "best_model": progress.get("best_model"),
        })
        complete_staging = True
        publish_path(staging, final)
        atomic_json(output / "latest.json", {"checkpoint": final.relative_to(output).as_posix()})
    finally:
        if staging.exists() and not complete_staging:
            cleanup_child(staging, checkpoints)
    saved = sorted(
        (path for path in checkpoints.glob("step-*") if (path / "state.pt").is_file()),
        key=lambda path: path.stat().st_mtime_ns, reverse=True,
    )
    for old in saved[2:]:
        cleanup_child(old, checkpoints)
    return final


def resolve_resume(value: str, output: Path) -> Path:
    candidate = output if value == "latest" else Path(value).resolve()
    if candidate.is_dir() and (candidate / "latest.json").exists():
        relative = json.loads((candidate / "latest.json").read_text(encoding="utf-8"))["checkpoint"]
        candidate = candidate / relative
    if candidate.is_dir():
        candidate = candidate / "state.pt"
    if not candidate.is_file():
        raise FileNotFoundError(f"Cannot locate resume checkpoint: {candidate}")
    return candidate


def export_best(model: SBERTNLI, output: Path, progress, config, hashes, score: float) -> str:
    root = output / "best_models"
    root.mkdir(exist_ok=True)
    name = f"step-{progress['global_step']:09d}-{uuid.uuid4().hex[:8]}"
    staging, final = root / (".tmp-" + name), root / name
    complete_staging = False
    try:
        model.export_sentence_transformer(staging, {
            "model_name": config["model_name"],
            "selection": {"split": "stsb-dev", "metric": "cosine_spearman",
                          "score": score, "global_step": progress["global_step"]},
            "training_provenance": {"config": config, "data_sha256": hashes},
        })
        complete_staging = True
        publish_path(staging, final)
    finally:
        if staging.exists() and not complete_staging:
            cleanup_child(staging, root)
    # Keep older best exports until the next complete checkpoint has committed.
    return final.relative_to(output).as_posix()


def clean_best_models(output: Path, progress: dict[str, Any]) -> None:
    """Retain exports referenced by either of the two restorable checkpoints."""
    keep = {progress.get("best_model")}
    for summary in (output / "checkpoints").glob("*/summary.json"):
        record = json.loads(summary.read_text(encoding="utf-8"))
        if record.get("best_model"):
            keep.add(record["best_model"])
    root = output / "best_models"
    if root.exists():
        for path in root.glob("step-*"):
            if path.relative_to(output).as_posix() not in keep:
                cleanup_child(path, root)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE / "data")
    parser.add_argument("--output-dir", type=Path, default=BASE / "runs" / "nli-bert-base")
    parser.add_argument("--model-name", default=str(BASE / "models" / "bert-base-uncased"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int, default=130)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0, help="Debugging cap on the global step; 0 means a complete run")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--resume", nargs="?", const="latest")
    args = parser.parse_args(argv)
    for name in ("batch_size", "eval_batch_size", "epochs", "max_seq_length", "threads", "save_steps", "eval_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_steps < 0 or args.learning_rate <= 0 or not 0 <= args.warmup_ratio <= 1:
        parser.error("Require max_steps >= 0, learning_rate > 0 and warmup_ratio in [0, 1]")
    return args


def _run(args: argparse.Namespace, stop_requested: list[bool]) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and ((output / "latest.json").exists() or (output / "sentence_transformer").exists()):
        raise FileExistsError(f"Output already contains a run: {output}; use --resume latest or another output directory")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    start = time.perf_counter()
    status = {"state": "loading", "pid": os.getpid(), "started_at": now(), "output_dir": str(output)}
    atomic_json(output / "status.json", {**status, "updated_at": now()})
    nli_path = select_data_file(args.data_dir, ("AllNLI-historical.tsv.gz", "AllNLI.tsv.gz", "allnli.tsv.gz", "AllNLI.tsv"))
    stsb_path = select_data_file(args.data_dir, ("stsbenchmark.tsv.gz", "stsb.tsv.gz", "stsbenchmark.tsv"))
    hashes = {"allnli": sha256(nli_path), "stsb": sha256(stsb_path)}
    examples = read_nli(nli_path)
    stsb = read_stsb(stsb_path)
    batches_per_epoch = math.ceil(len(examples) / args.batch_size)
    total_steps = batches_per_epoch * args.epochs
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)
    model_name = str(Path(args.model_name).resolve()) if Path(args.model_name).exists() else args.model_name
    model_source = None
    source_path = Path(model_name) / "source.json"
    if source_path.is_file():
        model_source = {"source_sha256": sha256(source_path),
                        "source": json.loads(source_path.read_text(encoding="utf-8"))}
        for filename, expected_hash in model_source["source"].get("sha256", {}).items():
            model_file = Path(model_name) / filename
            if model_file.resolve().parent != Path(model_name).resolve():
                raise ValueError(f"Invalid source manifest model filename: {filename}")
            if not model_file.is_file() or sha256(model_file) != expected_hash:
                raise ValueError(f"Pretrained model checksum mismatch: {model_file}")
    config = {
        "model_name": model_name, "batch_size": args.batch_size,
        "nli_filename": nli_path.name,
        "model_source": model_source,
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio, "warmup_steps": warmup_steps,
        "max_seq_length": args.max_seq_length, "threads": args.threads,
        "seed": args.seed, "total_steps": total_steps, "train_examples": len(examples),
        "eval_steps": args.eval_steps, "eval_batch_size": args.eval_batch_size,
        "optimizer": "AdamW(correct_bias=False,eps=1e-6,weight_decay=0.01)",
        "max_grad_norm": 1.0, "shuffle": "torch.randperm(seed+epoch)",
        "labels": LABELS, "pooling": "attention-masked-mean",
    }
    checkpoint = None
    if args.resume:
        # Training state is local and trusted, and includes Python/NumPy RNG state.
        checkpoint = torch.load(resolve_resume(args.resume, output), map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        if checkpoint["data_hashes"] != hashes:
            raise ValueError("Resume rejected: dataset SHA256 hashes changed")
        if checkpoint["config"] != config:
            differences = {key: (checkpoint["config"].get(key), value) for key, value in config.items()
                           if checkpoint["config"].get(key) != value}
            raise ValueError(f"Resume rejected: training configuration changed: {differences}")
    atomic_json(output / "run_config.json", {"config": config, "data_sha256": hashes})
    model = SBERTNLI(model_name, max_seq_length=args.max_seq_length).to("cpu")
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if name.endswith("bias") or "LayerNorm.weight" in name else decay).append(parameter)
    optimizer = OriginalAdamW([
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.learning_rate, eps=1e-6)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    progress = {
        "global_step": 0, "epoch": 0, "next_batch": 0,
        "best_dev_spearman": None, "best_model": None,
        "last_eval_step": -1, "train_seconds": 0.0, "last_loss": None,
        "completed": False, "test_spearman": None,
    }
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        progress = checkpoint["progress"]
        if progress["best_model"] and not (output / progress["best_model"]).is_dir():
            raise FileNotFoundError("The selected best model referenced by the checkpoint is missing")
        restore_rng(checkpoint["rng"])
        del checkpoint
    append_metric(output, {"event": "resume" if args.resume else "start", "global_step": progress["global_step"], "config": config})
    status.update({"total_steps": total_steps, "train_examples": len(examples), "batches_per_epoch": batches_per_epoch, "config": config})
    session_initial_step = progress["global_step"]
    session_initial_seconds = progress["train_seconds"]
    training_started = time.perf_counter()

    def write_status(state: str, **extra) -> None:
        elapsed = time.perf_counter() - training_started
        steps = progress["global_step"] - session_initial_step
        seconds_per_step = elapsed / steps if steps else None
        atomic_json(output / "status.json", {
            **status, **progress, "state": state, "updated_at": now(),
            "session_seconds": time.perf_counter() - start,
            "seconds_per_step": seconds_per_step,
            "estimated_remaining_seconds": seconds_per_step * (total_steps - progress["global_step"]) if seconds_per_step else None,
            **extra,
        })

    def persist(reason: str) -> None:
        write_status("saving", reason=reason)
        progress["train_seconds"] = session_initial_seconds + time.perf_counter() - training_started
        saved = save_checkpoint(output, model, optimizer, scheduler, progress, config, hashes)
        clean_best_models(output, progress)
        append_metric(output, {"event": "checkpoint", "reason": reason, "global_step": progress["global_step"], "path": str(saved)})

    def validate() -> None:
        write_status("evaluating_dev")
        score = evaluate_stsb(model, stsb["dev"], args.eval_batch_size)
        progress["last_eval_step"] = progress["global_step"]
        if progress["best_dev_spearman"] is None or score > progress["best_dev_spearman"]:
            best_path = export_best(model, output, progress, config, hashes, score)
            progress["best_model"] = best_path
            progress["best_dev_spearman"] = score
        append_metric(output, {"event": "stsb_dev", "global_step": progress["global_step"],
                               "cosine_spearman": score, "best_cosine_spearman": progress["best_dev_spearman"]})
        write_status("training")

    bar = tqdm(total=total_steps, initial=progress["global_step"], desc="AllNLI CPU", unit="step", mininterval=2)
    reason = "completed"
    try:
        model.train()
        if args.resume and not progress["completed"]:
            # Confirm checkpoint publication works before spending more CPU time.
            persist("resume_verified")
            write_status("training")
        while progress["epoch"] < args.epochs:
            if stop_requested[0] or (output / "STOP").exists():
                reason = "keyboard_interrupt" if stop_requested[0] else "stop_file"
                break
            if args.max_steps and progress["global_step"] >= args.max_steps:
                reason = "max_steps"
                break
            epoch = progress["epoch"]
            generator = torch.Generator().manual_seed(args.seed + epoch)
            order = torch.randperm(len(examples), generator=generator).tolist()
            for batch_index in range(progress["next_batch"], batches_per_epoch):
                if stop_requested[0] or (output / "STOP").exists():
                    reason = "keyboard_interrupt" if stop_requested[0] else "stop_file"
                    break
                if args.max_steps and progress["global_step"] >= args.max_steps:
                    reason = "max_steps"
                    break
                indices = order[batch_index * args.batch_size:(batch_index + 1) * args.batch_size]
                batch = [examples[index] for index in indices]
                first = model.tokenize([row[0] for row in batch], torch.device("cpu"))
                second = model.tokenize([row[1] for row in batch], torch.device("cpu"))
                labels = torch.tensor([row[2] for row in batch], dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(first, second), labels)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at step {progress['global_step']}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                progress["global_step"] += 1
                progress["next_batch"] = batch_index + 1
                progress["last_loss"] = float(loss.detach())
                progress["train_seconds"] = session_initial_seconds + time.perf_counter() - training_started
                bar.update(1)
                bar.set_postfix(loss=f"{progress['last_loss']:.4f}")
                append_metric(output, {"event": "train", "global_step": progress["global_step"],
                                       "epoch": epoch, "batch": batch_index, "loss": progress["last_loss"],
                                       "learning_rate": scheduler.get_last_lr()[0]})
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
        complete = progress["epoch"] >= args.epochs
        if complete:
            if progress["last_eval_step"] != progress["global_step"]:
                validate()
            # Only the dev-selected model is used for the one final test evaluation.
            best = output / progress["best_model"]
            final = output / "sentence_transformer"
            if not final.exists():
                staging = output / (".tmp-final-" + uuid.uuid4().hex)
                shutil.copytree(best, staging)
                publish_path(staging, final)
            if not progress["completed"]:
                write_status("evaluating_test")
                selected = SBERTNLI(str(final), max_seq_length=args.max_seq_length).to("cpu")
                progress["test_spearman"] = evaluate_stsb(selected, stsb["test"], args.eval_batch_size)
                del selected
                append_metric(output, {"event": "stsb_test", "global_step": progress["global_step"],
                                       "cosine_spearman": progress["test_spearman"], "model": str(final)})
                progress["completed"] = True
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
                      "global_step": progress["global_step"], "output_dir": str(output)}, ensure_ascii=False), flush=True)


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # SIGINT only requests a stop. A complete optimizer/scheduler step must finish
    # before its matching batch cursor and RNG can be checkpointed safely.
    stop_requested = [False]
    previous_handler = signal.getsignal(signal.SIGINT)
    def request_stop(_signum, _frame):
        stop_requested[0] = True
    with FileLock(str(output / ".training.lock"), timeout=0):
        signal.signal(signal.SIGINT, request_stop)
        try:
            _run(args, stop_requested)
        finally:
            signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as error:
        print(f"Training failed: {error}", file=sys.stderr, flush=True)
        raise
