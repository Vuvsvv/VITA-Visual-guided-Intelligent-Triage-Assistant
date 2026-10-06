param(
    [switch]$ListTasksOnly
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Android = Join-Path $Root "android"
$Gradle = Join-Path $Android "gradlew.bat"

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

if (-not (Test-Path $Gradle)) {
    throw "Cannot find Gradle wrapper at $Gradle"
}

Push-Location $Android
try {
    Write-Host "Discovering Gradle tasks..."
    $tasks = & .\gradlew.bat tasks --all
    if ($LASTEXITCODE -ne 0) {
        throw "Gradle task discovery failed with exit code $LASTEXITCODE"
    }
    $tasks | Set-Content -Encoding UTF8 (Join-Path $Root "docs\ANDROID_GRADLE_TASKS.txt")

    if ($ListTasksOnly) {
        $tasks
        return
    }

    if ($tasks -match "testDebugUnitTest") {
        Write-Host "Running Android unit tests: testDebugUnitTest"
        Invoke-Checked { & .\gradlew.bat testDebugUnitTest } "Android unit tests"
    }
    elseif ($tasks -match "(^|\s)test($|\s)") {
        Write-Host "testDebugUnitTest not available; running closest test task: test"
        Invoke-Checked { & .\gradlew.bat test } "Android unit tests"
    }
    else {
        Write-Host "No Android unit test task found. Available tasks were written to docs\ANDROID_GRADLE_TASKS.txt"
        throw "Android unit test task is unavailable."
    }

    if ($tasks -match "assembleDebug") {
        Write-Host "Running Android debug build: assembleDebug"
        Invoke-Checked { & .\gradlew.bat assembleDebug } "Android debug build"
    }
    else {
        Write-Host "assembleDebug not found. Available tasks were written to docs\ANDROID_GRADLE_TASKS.txt"
        throw "Android assembleDebug task is unavailable."
    }
}
finally {
    Pop-Location
}
