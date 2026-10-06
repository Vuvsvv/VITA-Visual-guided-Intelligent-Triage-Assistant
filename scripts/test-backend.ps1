param(
    [switch]$SkipInstall
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

function Invoke-PythonSnippet {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Code,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    $TempFile = Join-Path $Backend ("__baseline_check_{0}.py" -f ([System.Guid]::NewGuid().ToString("N")))
    try {
        Set-Content -LiteralPath $TempFile -Value $Code -Encoding UTF8
        Invoke-Checked { & $Python $TempFile } $Description
    }
    finally {
        if (Test-Path $TempFile) {
            Remove-Item -LiteralPath $TempFile -Force
        }
    }
}

if (-not (Test-Path $Python) -or -not $SkipInstall) {
    & (Join-Path $PSScriptRoot "setup.ps1") -SkipBackendInstall:$SkipInstall
}

Push-Location $Backend
try {
    Write-Host "Running backend unittest suite..."
    Invoke-Checked { & $Python -m unittest tests.test_backend_flow } "Backend unittest suite"

    Write-Host "Running department AI adapter unittest suite..."
    Invoke-Checked { & $Python -m unittest tests.test_project_smart_department_adapter } "Department AI adapter unittest suite"

    Write-Host "Running schedule filtering unittest suite..."
    Invoke-Checked { & $Python -m unittest tests.test_schedule_filter } "Schedule filtering unittest suite"

    Write-Host "Running specialty scoring unittest suite..."
    Invoke-Checked { & $Python -m unittest tests.test_specialty_scoring } "Specialty scoring unittest suite"

    Write-Host "Running selective Wu merge unittest suite..."
    Invoke-Checked { & $Python -m unittest tests.test_selective_wu_merge } "Selective Wu merge unittest suite"

    Write-Host "Running recommendation data access unittest suite..."
    Invoke-Checked { & $Python -m unittest tests.test_recommendation_data_access } "Recommendation data access unittest suite"

    Write-Host "Running appointment ranking unittest suite..."
    Invoke-Checked { & $Python -m unittest tests.test_appointment_ranking } "Appointment ranking unittest suite"

    Write-Host "Running follow-up recommendation unittest suite..."
    Invoke-Checked { & $Python -m unittest tests.test_followup_service } "Follow-up recommendation unittest suite"

    Write-Host "Running quick-search unittest suite..."
    Invoke-Checked { & $Python -m unittest tests.test_quick_search } "Quick-search unittest suite"

    Write-Host "Running full backend pytest suite..."
    Invoke-Checked { & $Python -m pytest tests -q } "Backend pytest suite"

    Write-Host "Validating /health and OpenAPI routes..."
    $contractCheck = @'
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
health = client.get("/health")
assert health.status_code == 200, health.text
assert health.json() == {"status": "ok"}, health.text

openapi = client.get("/openapi.json")
assert openapi.status_code == 200, openapi.text
paths = openapi.json().get("paths", {})
required = [
    "/chat",
    "/recommend",
    "/generate_script",
    "/voice/chat",
    "/voice/tts",
    "/health",
    "/followup/recommend",
    "/schedules/search",
]
missing = [path for path in required if path not in paths]
assert not missing, f"Missing OpenAPI paths: {missing}"
print("Backend health and OpenAPI contract checks passed.")
'@
    Invoke-PythonSnippet $contractCheck "Backend health and OpenAPI contract checks"
}
finally {
    Pop-Location
}
