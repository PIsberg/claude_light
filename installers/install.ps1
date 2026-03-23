# Claude Light installer -- Windows (PowerShell 5.1+)
# Run from the claude_light directory:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\install.ps1
#Requires -Version 5.1

$ErrorActionPreference = "Stop"

function Info  { Write-Host "[install] $args" -ForegroundColor Green }
function Warn  { Write-Host "[warn]    $args" -ForegroundColor Yellow }
function Fail  { Write-Host "[error]   $args" -ForegroundColor Red; exit 1 }

# --- Python ------------------------------------------------------------------
$python = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ok = & $cmd -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $python = $cmd; break }
    } catch {}
}
if (-not $python) { Fail "Python 3.9+ is required but was not found. Install from https://python.org" }
Info "Using $(&$python --version)"

# --- Required packages -------------------------------------------------------
Info "Installing required packages (sentence-transformers pulls PyTorch ~1.5 GB on first run)..."
& $python -m pip install --upgrade sentence-transformers numpy watchdog anthropic prompt_toolkit
if ($LASTEXITCODE -ne 0) { Fail "Failed to install required packages." }

# --- Optional packages -------------------------------------------------------
Info "Installing optional packages (tree-sitter, rich, einops)..."
$optional = @(
    "tree-sitter",
    "tree-sitter-java",
    "tree-sitter-python",
    "tree-sitter-go",
    "tree-sitter-rust",
    "tree-sitter-javascript",
    "tree-sitter-typescript",
    "rich",
    "einops"
)
& $python -m pip install --upgrade $optional
if ($LASTEXITCODE -ne 0) {
    Warn "Some optional packages failed - the tool will still work, with reduced functionality."
}

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
