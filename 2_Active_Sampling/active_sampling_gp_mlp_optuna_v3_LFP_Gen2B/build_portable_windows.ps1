param(
    [string]$AppVersion = "0.1.0",
    [switch]$Sign,
    [switch]$Lean,
    [switch]$Console,
    [switch]$OneFile,
    [string]$CertPath = "",
    [string]$TimestampUrl = "http://timestamp.digicert.com"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$distBackup = Join-Path $projectRoot "dist_backup_prev"
$buildBackup = Join-Path $projectRoot "build_backup_prev"

if (Test-Path $distBackup) { Remove-Item $distBackup -Recurse -Force }
if (Test-Path $buildBackup) { Remove-Item $buildBackup -Recurse -Force }

if (Test-Path "dist") { Move-Item "dist" $distBackup }
if (Test-Path "build") { Move-Item "build" $buildBackup }

$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Python not found: $pythonExe"
}

$bundlePath = Join-Path (Split-Path $projectRoot -Parent) "active_sampling_gp_mlp_optuna_v3_LFP_Gen2B_SC40\outputs\latest_surrogate_bundle.pkl"
if (-not (Test-Path $bundlePath)) {
    throw "Embedded bundle not found: $bundlePath"
}

& $pythonExe -m pip install --upgrade pip pyinstaller

try {

$pyiArgs = @(
    "--noconfirm",
    "--clean",
    (Join-Path $projectRoot "SurrogatePredictor.spec"),
    "--"
)

if (-not $Console) {
    $pyiArgs += "--windowed"
}

if (-not $OneFile) {
    $pyiArgs += "--onedir"
}

& $pythonExe -m PyInstaller @pyiArgs
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

if ($Lean) {
    Write-Host "[INFO] Runtime-only data collection is enabled for all builds."
}

if ($OneFile) {
    $exePath = Join-Path $projectRoot "dist\SurrogatePredictor.exe"
    if (-not (Test-Path $exePath)) {
        throw "Build output EXE not found: $exePath"
    }
}
else {
    $distRoot = Join-Path $projectRoot "dist\SurrogatePredictor"
    if (-not (Test-Path $distRoot)) {
        throw "Build output folder not found: $distRoot"
    }

    # Keep package path short to avoid Windows long-path issues during copy.
    Copy-Item "README_PORTABLE_DEPLOY.md" (Join-Path $distRoot "README_PORTABLE_DEPLOY.md") -Force
}

if ($Sign) {
    if ([string]::IsNullOrWhiteSpace($CertPath)) {
        throw "-Sign was used but -CertPath is empty."
    }
    if (-not $OneFile) {
        $exePath = Join-Path $distRoot "SurrogatePredictor.exe"
    }
    if (-not (Test-Path $exePath)) {
        throw "EXE not found for signing: $exePath"
    }

    $signtool = "signtool.exe"
    & $signtool sign /fd SHA256 /td SHA256 /tr $TimestampUrl /f $CertPath $exePath
}

if ($OneFile) {
    $stageDir = Join-Path $projectRoot "dist\SP"
    if (Test-Path $stageDir) { Remove-Item $stageDir -Recurse -Force }
    New-Item -ItemType Directory -Path $stageDir -Force | Out-Null
    Copy-Item $exePath (Join-Path $stageDir "SurrogatePredictor.exe") -Force
    Copy-Item "README_PORTABLE_DEPLOY.md" (Join-Path $stageDir "README_PORTABLE_DEPLOY.md") -Force

    $zipPath = Join-Path $projectRoot ("dist\SP_v" + $AppVersion + ".zip")
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Compress-Archive -Path (Join-Path $stageDir "*") -DestinationPath $zipPath
}
else {
    $zipPath = Join-Path $projectRoot ("dist\SurrogatePredictor_portable_v" + $AppVersion + ".zip")
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Compress-Archive -Path (Join-Path $distRoot "*") -DestinationPath $zipPath
}

Write-Host "[OK] Portable package: $zipPath"

if (Test-Path $distBackup) { Remove-Item $distBackup -Recurse -Force }
if (Test-Path $buildBackup) { Remove-Item $buildBackup -Recurse -Force }
}
catch {
    if ((-not (Test-Path "dist")) -and (Test-Path $distBackup)) {
        Move-Item $distBackup "dist"
    }
    if ((-not (Test-Path "build")) -and (Test-Path $buildBackup)) {
        Move-Item $buildBackup "build"
    }
    throw
}
