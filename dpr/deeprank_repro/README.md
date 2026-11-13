# Deeprank DPR implementation snapshot

This directory mirrors the DPR and NFCorpus retrieval code from
`Q:\repos\deeprank` on branch `codex/dpr_repro`, copied on 2026-11-13.
The 32 copied source files retain their original repository-relative paths
under `lightning/` and `smile/`; each was checked against its source with
SHA256 after copying. The original paper-era snapshot remains separately in
`../official/`, and the small standalone teaching implementation remains in
`../core.py`.

## Layout

- `lightning/src/smile_lightning/apps/retrieval/dpr/`: NQ DPR data loader, two
  independent BERT encoders, and Lightning training module.
- `lightning/src/smile_lightning/apps/retrieval/nfcorpus/`: NFCorpus data loader
  and NQ-initialized fine-tuning module. Its `dot` mode is the original
  unnormalized scoring path; `cosine` normalizes CLS vectors and uses training
  temperature 0.05 (`score_scale=20`).
- `lightning/conf/apps/retrieval/`: Lightning application settings.
- `lightning/scripts/retrieval/`: NFCorpus test evaluator and dev-only dot versus
  cosine diagnostic. The evaluator reads `similarity_mode` from the checkpoint.
- `lightning/tests/`: focused data, model, scoring, and evaluator tests.
- `smile/src/smile/pipelines/retrieval/`: NQ training and NFCorpus fine-tuning
  pipeline definitions and configs, including `nfcorpus_finetune/config/cosine.yaml`.

## Runtime boundary

This is a source snapshot, **not an independent SMILE or Lightning project**.
The pipeline imports the full `smile` framework and the Lightning app imports
`smile_lightning.core`, which are supplied by the Deeprank checkout. Submit
AML jobs from `Q:\repos\deeprank\smile` with its configured Conda environment.
The target repository's `dpr/.venv` is only for the older PyTorch smoke test;
it lacks the dependencies required by this snapshot.

The copied test evaluator can run locally from `Q:\repos\ranking_retrieval`
with `sbert/.venv`, which contains PyTorch, Transformers, and SciPy. It loads
the sibling `bm25/` metric and `sbert/` raw-text loader. Supply an existing
checkpoint and a fresh output filename, for example:

```powershell
& .\sbert\.venv\Scripts\python.exe .\dpr\deeprank_repro\lightning\scripts\retrieval\evaluate_dpr_nfcorpus.py `
  --checkpoint 'Q:\repos\deeprank\lightning\outputs\dpr_nfcorpus_cosine_c70c1303\best.ckpt' `
  --data-dir 'Q:\repos\ranking_retrieval\nfcorpus' `
  --output 'Q:\repos\ranking_retrieval\dpr\outputs\cosine-test.json' `
  --source-aml-run-id c70c1303-4d7b-4afb-9100-cacde1863193 `
  --training-data 'NQ warm start; NFCorpus train cosine fine-tuning' `
  --tokenizer 'Q:\repos\ranking_retrieval\sbert\models\bert-base-uncased' `
  --text-mode raw-chunks --batch-size 16 --threads 8 --max-length 256 --overlap 64
```

For model and pipeline tests, use the complete Deeprank checkout and its
`lightning/tests/apps/retrieval/` and `smile/tests/pipelines/retrieval/`
test commands. The mirror preserves those test sources for review; it does not
vendor their framework dependencies or the multi-gigabyte AML checkpoints.
The [parent README](../README.md) records the experiment sequence, Run IDs,
evaluation protocol, and measured results.
