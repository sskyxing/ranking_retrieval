param(
    [string]$OutputDir = (Join-Path $PSScriptRoot 'runs/nli-bert-base'),
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$train = Join-Path $PSScriptRoot 'train.py'
if (-not (Test-Path -LiteralPath $python)) { throw 'Create sbert/.venv and install requirements first.' }
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$processPath = Join-Path $OutputDir 'process.json'
if (Test-Path -LiteralPath $processPath) {
    $previous = Get-Content -Raw -LiteralPath $processPath | ConvertFrom-Json
    $active = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $previous.pid)
    if ($active -and $active.CommandLine -and $active.CommandLine.Contains($train)) {
        throw ('Training is already running with PID ' + $previous.pid)
    }
}
if (Test-Path -LiteralPath (Join-Path $OutputDir 'STOP')) {
    throw 'Remove the STOP file explicitly before restarting training.'
}
if (-not $Resume -and (Test-Path -LiteralPath (Join-Path $OutputDir 'status.json'))) {
    throw 'Existing run found. Use -Resume or a fresh -OutputDir.'
}
$arguments = @(
    '-u', ('"' + $train + '"'),
    '--data-dir', ('"' + (Join-Path $PSScriptRoot 'data') + '"'),
    '--model-name', ('"' + (Join-Path $PSScriptRoot 'models/bert-base-uncased') + '"'),
    '--output-dir', ('"' + $OutputDir + '"'),
    '--threads', '8', '--batch-size', '16', '--epochs', '1',
    '--max-seq-length', '130', '--seed', '42', '--save-steps', '500', '--eval-steps', '1000'
)
if ($Resume) { $arguments += @('--resume', 'latest') }
$timestamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$stdout = Join-Path $OutputDir ('train.' + $timestamp + '.stdout.log')
$stderr = Join-Path $OutputDir ('train.' + $timestamp + '.stderr.log')
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$processTempPath = $processPath + '.tmp'
[ordered]@{
    pid = $process.Id
    started_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    output_dir = $OutputDir
    stdout = $stdout
    stderr = $stderr
    executable = $python
    arguments = $arguments
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $processTempPath -Encoding utf8
Move-Item -LiteralPath $processTempPath -Destination $processPath -Force
Start-Sleep -Seconds 3
$process.Refresh()
if ($process.HasExited) {
    Get-Content -LiteralPath $stderr -Tail 30
    throw ('Training exited immediately, exit code ' + $process.ExitCode)
}
Write-Output ('Training started. PID: ' + $process.Id)
Write-Output ('Status: ' + (Join-Path $OutputDir 'status.json'))
Write-Output ('Stdout: ' + $stdout)
Write-Output ('Stderr: ' + $stderr)
