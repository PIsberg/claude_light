# Generate CycloneDX SBOM for claude-light
# Requires: pip install cyclonedx-bom

$ErrorActionPreference = "Stop"

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = (Get-Item "$SCRIPT_DIR/..").FullName

Write-Host "[sbom] Generating CycloneDX SBOM..." -ForegroundColor Green

if (-not (Get-Command "cyclonedx-py" -ErrorAction SilentlyContinue)) {
    Write-Host "[sbom] cyclonedx-py not found. Installing..." -ForegroundColor Yellow
    & pip install cyclonedx-bom --quiet
}

Set-Location $PROJECT_ROOT
& cyclonedx-py requirements -i requirements.txt -o sbom.json

Write-Host "[sbom] SBOM generated at $PROJECT_ROOT\sbom.json" -ForegroundColor Green
