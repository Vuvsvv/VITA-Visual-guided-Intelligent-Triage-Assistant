param(
    [int]$Port = 8000,
    [string]$HostName = "0.0.0.0"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Python = Join-Path $Backend ".venv\Scripts\python.exe"

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

if (-not (Test-Path $Python)) {
    & (Join-Path $PSScriptRoot "setup.ps1")
}

Push-Location $Backend
try {
    Invoke-Checked { & $Python -m uvicorn app.main:app --host $HostName --port $Port } "Starting backend"
}
finally {
    Pop-Location
}
