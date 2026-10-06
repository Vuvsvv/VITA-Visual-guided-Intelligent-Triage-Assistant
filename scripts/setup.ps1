param(
    [switch]$SkipBackendInstall
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Venv = Join-Path $Backend ".venv"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [ScriptBlock]$Command,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $Venv)) {
    Write-Host "Creating backend virtual environment..."
    Invoke-Checked { python -m venv $Venv } "Creating backend virtual environment"
}

$Python = Join-Path $Venv "Scripts\python.exe"

if (-not $SkipBackendInstall) {
    Write-Host "Installing backend development requirements..."
    Invoke-Checked { & $Python -m pip install --upgrade pip } "Upgrading pip"
    Invoke-Checked { & $Python -m pip install -r (Join-Path $Backend "requirements-dev.txt") } "Installing backend development requirements"
}

Write-Host "Setup complete."
