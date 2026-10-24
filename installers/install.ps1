# Claude Light installer -- Windows (PowerShell 5.1+)
# Run from the claude_light directory:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\installers\install.ps1
#Requires -Version 5.1

$ErrorActionPreference = "Stop"

function Info  { Write-Host "[install] $args" -ForegroundColor Green }
function Warn  { Write-Host "[warn]    $args" -ForegroundColor Yellow }
function Fail  { Write-Host "[error]   $args" -ForegroundColor Red; exit 1 }

# --- Python ------------------------------------------------------------------
$python = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ok = & $cmd -c "import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $python = $cmd; break }
    } catch {}
}
if (-not $python) { Fail "Python 3.12+ is required but was not found. Install from https://python.org" }
Info "Using $(&$python --version)"

# --- Packages ----------------------------------------------------------------
# requirements.txt is the one list of versions (the README and CI read it too).
# sentence-transformers pulls PyTorch, about 1.5 GB on first run.
Info "Installing packages from requirements.txt (sentence-transformers pulls PyTorch ~1.5 GB on first run)..."
$requirements = Join-Path (Split-Path -Parent $PSScriptRoot) "requirements.txt"
& $python -m pip install --upgrade -r $requirements
if ($LASTEXITCODE -ne 0) { Fail "Failed to install packages from requirements.txt." }

# --- API key check -----------------------------------------------------------
Write-Host ""
$key = [System.Environment]::GetEnvironmentVariable("ANTHROPIC_API_KEY", "User")
if (-not $key) {
    $key = [System.Environment]::GetEnvironmentVariable("ANTHROPIC_API_KEY", "Machine")
}
if (-not $key) {
    Warn "ANTHROPIC_API_KEY is not set."
    Write-Host "  Set it as a persistent user environment variable, e.g.:"
    Write-Host '  [System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY","sk-ant-...","User")'
    Write-Host "  Or place it in a .env file in your project directory."
} else {
    Info "ANTHROPIC_API_KEY is already set."
}

Write-Host ""
Info "Installation complete. Run the tool from your project root:"
Write-Host "  python claude_light.py"
