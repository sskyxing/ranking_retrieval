param(
    [string]$OutputDir = (Join-Path $PSScriptRoot 'runs/nfcorpus-retrieval'),
    [string]$ModelDir = (Join-Path $PSScriptRoot 'models/nli-step3000-nfcorpus-init'),
    [ValidateRange(1,100)][int]$Epochs = 3,
    [ValidateRange(2,512)][int]$MaxSeqLength = 130,
    [ValidateSet('query','all-positives')][string]$Sampling = 'query',
    [ValidateRange(1,10000)][int]$SaveSteps = 100,
    [ValidateRange(1,10000)][int]$EvalSteps = 100,
    [switch]$RawChunks,
    [ValidateRange(0,510)][int]$ChunkOverlap = 64,
    [ValidateRange(1,128)][int]$EncoderMicroBatch = 8,
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$train = Join-Path $PSScriptRoot 'train_retrieval.py'
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
$ModelDir = [IO.Path]::GetFullPath($ModelDir)
if (-not (Test-Path -LiteralPath $python)) { throw 'Create sbert/.venv and install requirements first.' }
if (-not (Test-Path -LiteralPath (Join-Path $ModelDir 'sbert_config.json'))) { throw 'A local SBERT training export is required.' }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$processPath = Join-Path $OutputDir 'process.json'
if (Test-Path -LiteralPath $processPath) {
    $previous = Get-Content -Raw -LiteralPath $processPath | ConvertFrom-Json
    $active = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $previous.pid)
    if ($active -and $active.CommandLine -and $active.CommandLine.Contains($train)) {
        throw ('Retrieval training is already running with PID ' + $previous.pid)
    }
}
if (Test-Path -LiteralPath (Join-Path $OutputDir 'STOP')) { throw 'Remove STOP explicitly before restarting.' }
if (-not $Resume -and (Test-Path -LiteralPath (Join-Path $OutputDir 'status.json'))) { throw 'Existing run: use -Resume or a fresh -OutputDir.' }
$arguments = @(
    '-u', ('"' + $train + '"'),
    '--data-dir', ('"' + (Join-Path $PSScriptRoot '../nfcorpus') + '"'),
    '--model-dir', ('"' + $ModelDir + '"'),
    '--output-dir', ('"' + $OutputDir + '"'),
    '--cache-path', ('"' + (Join-Path $PSScriptRoot 'data/nfcorpus-hard-negatives.json') + '"'),
    '--threads', '8', '--batch-size', '16', '--epochs', $Epochs.ToString(),
    '--max-seq-length', $MaxSeqLength.ToString(), '--sampling', $Sampling,
    '--temperature', '0.05', '--hard-negative-count', '20',
    '--learning-rate', '0.00002', '--seed', '42', '--save-steps', $SaveSteps.ToString(), '--eval-steps', $EvalSteps.ToString()
)
if ($RawChunks) {
    $arguments += @('--raw-chunks', '--chunk-overlap', $ChunkOverlap.ToString(),
                    '--encoder-micro-batch', $EncoderMicroBatch.ToString())
    $cacheFlag = [array]::IndexOf($arguments, '--cache-path')
    $arguments[$cacheFlag + 1] = ('"' + (Join-Path $PSScriptRoot 'data/nfcorpus-raw-hard-negatives.json') + '"')
}
if ($Resume) { $arguments += @('--resume', 'latest') }
$timestamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$stdout = Join-Path $OutputDir ('train.' + $timestamp + '.stdout.log')
$stderr = Join-Path $OutputDir ('train.' + $timestamp + '.stderr.log')
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$temporary = $processPath + '.tmp'
[ordered]@{pid=$process.Id; started_at_utc=(Get-Date).ToUniversalTime().ToString('o');
    output_dir=$OutputDir; model_dir=$ModelDir; stdout=$stdout; stderr=$stderr;
    executable=$python; arguments=$arguments} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $temporary -Encoding utf8
Move-Item -LiteralPath $temporary -Destination $processPath -Force
Start-Sleep -Seconds 3
$process.Refresh()
if ($process.HasExited) {
    Get-Content -LiteralPath $stderr -Tail 30
    throw ('Retrieval training exited, code ' + $process.ExitCode)
}
Write-Output ('Retrieval training started. Launcher PID: ' + $process.Id)
Write-Output ('Status: ' + (Join-Path $OutputDir 'status.json'))
Write-Output ('Stdout: ' + $stdout)
Write-Output ('Stderr: ' + $stderr)
