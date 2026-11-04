param(
    [string]$OutputDir = (Join-Path $PSScriptRoot 'runs/nfcorpus-retrieval-256-full'),
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'start_retrieval_training.ps1') `
    -OutputDir $OutputDir `
    -ModelDir (Join-Path $PSScriptRoot 'models/nli-step3000-nfcorpus-init') `
    -Epochs 1 -MaxSeqLength 256 -Sampling 'all-positives' -SaveSteps 250 -EvalSteps 500 `
    -Resume:$Resume
