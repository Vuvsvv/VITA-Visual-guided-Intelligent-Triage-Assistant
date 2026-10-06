$ErrorActionPreference = "Stop"

& (Join-Path $PSScriptRoot "test-backend.ps1")
if ($LASTEXITCODE -ne 0) {
    throw "Backend baseline checks failed with exit code $LASTEXITCODE"
}

& (Join-Path $PSScriptRoot "test-android.ps1")
if ($LASTEXITCODE -ne 0) {
    throw "Android baseline checks failed with exit code $LASTEXITCODE"
}

Write-Host "All baseline checks completed."
