# NFCorpus DPR Fine-Tuning

This pipeline adapts the already NQ-trained DPR bi-encoder to NFCorpus. It
starts from the NQ question/context tower weights, trains with NFCorpus
relevance judgments and a staged BM25 hard-negative cache, and saves
checkpoints for each epoch. The Lightning app chooses its best checkpoint by
full dev-corpus Recall@10. The independent NQ training pipeline is unchanged.

## Usage

The default config points to one per-user development folder in the
`relevance_c09` datastore. Validate the graph before submitting:

```bash
cd smile
conda run -n smile-codex-dpr-repro python run_smile.py retrieval.nfcorpus_finetune -v -ng
conda run -n smile-codex-dpr-repro python run_smile.py retrieval.nfcorpus_finetune -ng
```

For the optional normalized CLS experiment, select the `cosine` config:

```bash
conda run -n smile-codex-dpr-repro python run_smile.py retrieval.nfcorpus_finetune -c cosine -v -ng
conda run -n smile-codex-dpr-repro python run_smile.py retrieval.nfcorpus_finetune -c cosine -ng
```

This scenario retains the same NQ checkpoint, NFCorpus train/dev data,
batch size, learning rate, and five epochs as `default.yaml`. It normalizes
the CLS embeddings for both training and dev retrieval, and scales training
logits by 20 (temperature 0.05). Its experiment name and checkpoint output
path are separate from the dot-product run. The existing NQ DPR checkpoint
is still only the initialization; this scenario does not resume the earlier
NFCorpus fine-tuning optimizer or weights. Select the new best checkpoint
using dev Recall@10 before running any test evaluation. The standalone
`evaluate_dpr_nfcorpus.py` script currently scores raw dot products only;
extend it to cosine scoring before using it on a cosine-trained checkpoint.

To change the staged data or NQ checkpoint, edit the two ports under
`pipeline.inputs` in `config/default.yaml`. `nfcorpus_data` must end with `/`
so SMILE treats it as a folder. `nq_checkpoint` is a single `.ckpt` file.

## Pipeline Steps

```text
Step 1: smile_lightning_train (distributed GPU) — warm-start and fine-tune DPR
```

### Data Flow

```text
NFCorpus train/dev folder ─┐
                          ├─> NFCorpusDataModule + DPR towers ─> checkpoints
NQ best.ckpt ──────────────┘
```

The data folder needs `train.docs`, `train.titles.queries`,
`train.2-1-0.qrel`, the matching `dev.*` files, `raw/doc_dump.txt`,
`raw/nfdump.txt`, and `hard_negatives_raw_train.json`. No test split is used
for training or checkpoint choice. The fine-tuning code filters train queries
whose text overlaps the dev queries.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nfcorpus_data` | per-user NFCorpus folder | One read-only AML input folder |
| `nq_checkpoint` | NQ `best.ckpt` | Read-only initial tower weights |
| `epochs` | 5 | Fine-tuning epochs |
| `per_device_batch_size` | 8 | Queries per GPU |
| `learning_rate` | 1e-5 | Fresh AdamW learning rate |
| `warmup_steps` | 10 | Fresh optimizer warmup steps |
| `hard_negative_count` | 20 | BM25 candidate negatives per train query |
| `hard_negatives_per_query` | 1 | Exactly one candidate is supported and sampled per positive pair |
| `max_context_length` / `context_overlap` | 256 / 64 | Passage window and overlap |
| `similarity_mode` | dot | `dot` preserves DPR scoring; `cosine` L2-normalizes each CLS embedding |
| `score_scale` | 1.0 | Training logit multiplier; cosine scenario uses 20 for temperature 0.05 |
| `config_additional_override` | empty | Advanced Hydra overrides appended last |

The job uses one `ND96isr_MI300X_v5` node with eight MI300X GPUs in the
existing Singularity VC, FP32/DDP, and a per-user development output path.
The model's score scaling defaults to 1.0, matching the NQ raw dot product.
Training keeps the final partial batch so every eligible positive pair is used
in each epoch. `context_overlap` must be smaller than
`max_context_length - 3`, leaving room for BERT special tokens.
For a fair comparison with BM25 and SBERT, evaluate the selected checkpoint
on the untouched NFCorpus test split using the same corpus and metrics.
The evaluator requires an explicit training-data description and accepts an
optional source AML run ID so its report does not inherit stale provenance:

```bash
python lightning/scripts/retrieval/evaluate_dpr_nfcorpus.py \
  --checkpoint <best.ckpt> \
  --data-dir <ranking_retrieval/nfcorpus> \
  --output <results.json> \
  --source-aml-run-id <fine-tune-run-id> \
  --training-data "Natural Questions warm start; NFCorpus train fine-tuning"
```

## Known Issues

- An AML job has not been submitted merely by validating this pipeline.
- The checkpoint input must be compatible with the NQ DPR tower architecture.
