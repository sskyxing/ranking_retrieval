# Dense Passage Retrieval Training

This pipeline reproduces the Natural Questions bi-encoder training stage from
*Dense Passage Retrieval for Open-Domain Question Answering* (EMNLP 2020). It
mounts DPR-format train/dev JSON arrays and delegates the single GPU training
step to the shared `smile_lightning.train` pipeline. It does not build the
passage index or run end-to-end QA evaluation.

Reference implementation: `Q:\repos\ranking_retrieval\dpr` (read-only; not
packaged into this pipeline).

## Usage

The default config points to the official NQ train/dev JSON files staged under
`/local/users/sixing/dpr/nq-emnlp2020/` in datastore `relevance_c09`.
Validate before submitting:

```bash
cd smile
conda run -n smile-<branch> python run_smile.py retrieval.dpr_training -v
conda run -n smile-<branch> python run_smile.py retrieval.dpr_training
```

The committed default is a per-user development run. No production output path
is configured. To change only the data, override the data ports directly:

```bash
conda run -n smile-<branch> python run_smile.py retrieval.dpr_training -v \
  pipeline.inputs.train_data=azureml://datastores/relevance_c09/paths/<train.json> \
  pipeline.inputs.validation_data=azureml://datastores/relevance_c09/paths/<dev.json>
```

For a cheap configuration/data smoke test, use the CPU scenario and small JSON
files. It is intentionally not a paper-scale run:

```bash
conda run -n smile-<branch> python run_smile.py retrieval.dpr_training -c smoke -v
```

## Pipeline Steps

```text
Step 1: smile_lightning_train (DistributedComponent) — train and validate DPR
```

### Data Flow

```text
NQ train JSON ─┐
               ├─> DPRDataModule ─> DPRLightningModule (DDP) ─> checkpoints
NQ dev JSON ───┘
```

Each JSON file must be a top-level array of DPR examples containing `question`,
`positive_ctxs`, and `hard_negative_ctxs`; `negative_ctxs` is optional but is
used by the paper-style validation pool when present. Each context needs string
`title` and `text` fields. Training uses the first positive and a sampled hard
negative; validation uses a fixed passage pool for average-rank evaluation.

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_data` | placeholder URI | DPR NQ training JSON pipeline input |
| `validation_data` | placeholder URI | DPR NQ dev JSON pipeline input |
| `model_name_or_path` | `bert-base-uncased` | Hugging Face name or a path available inside the job |
| `epochs` | `40` | Training epochs from the released DPR NQ recipe |
| `per_device_batch_size` | `16` | Questions per GPU; 8 GPUs give global batch 128 |
| `learning_rate` | `2e-5` | Released DPR code/config value; the paper text states `1e-5` |
| `warmup_steps` | `1237` | Linear warmup optimizer steps |
| `seq_length` | `256` | Maximum length for both question and passage |
| `workers` | `4` | DataLoader workers per process |
| `seed` | `12345` | BaseApp and DataLoader seed |
| `validation_num_hard_negatives` | `30` | Hard negatives per validation question pool |
| `validation_num_other_negatives` | `30` | Ordinary negatives per validation question pool |
| `validation_max_questions` | `10000` | Maximum validation questions for average rank |
| `config_additional_override` | empty | Advanced Hydra overrides appended last |

The formal configuration uses workspace/compute `wxtcssingwestus3`, datastore
`relevance_c09`, Singularity VC `webxtcoresearchwus3`, one
`ND96isr_MI300X_v5` node (8 GPUs), `Premium`/`Low`, and non-preemptible
distributed training with `canadacentral` as the requested location. Precision
is exact FP32 (`32-true`), gradients are
clipped to a norm of `2.0`, and the strategy is DDP—not BF16 or DeepSpeed.
The app selects the slow BERT tokenizer and forces every token type ID to zero
to match the paper-era DPR tensorizer.
It retains every epoch checkpoint because the released implementation switches
its selection metric from NLL to average rank at epoch 30.
The MI300X resource choice comes from the user-supplied reference branch; the
linked AML run itself could not be queried locally because Azure CLI was not
authenticated, so run-time submission overrides are not asserted here.

### Config Files

| File | Purpose |
|------|---------|
| `config/default.yaml` | Paper-scale GPU development configuration |
| `config/smoke.yaml` | One-process CPU smoke configuration |
| `smile_lightning/train/config/retrieval/dpr/*.yaml` | Low-level direct shared-trainer scenarios |
| `lightning/conf/apps/retrieval/dpr/train_app.yaml` | DPR Hydra object graph and callbacks |
| `lightning/conf/apps/retrieval/dpr/trainer/gpu.yaml` | Eight-GPU FP32/DDP trainer |
| `lightning/conf/apps/retrieval/dpr/trainer/cpu.yaml` | Small local/AML CPU trainer |

## Known Issues

- The placeholder inputs intentionally fail dataset validation until replaced.
- `bert-base-uncased` requires network/cache access unless
  `model_name_or_path` points to a mounted local model directory.
- The command wrapper requires a mounted model path without whitespace.
- The full DPR JSON arrays are loaded into each worker process; plan memory and
  `workers` accordingly.
