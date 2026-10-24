#!/usr/bin/env bash
# Claude Light installer — Linux / macOS
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

info()  { echo -e "${GREEN}[install]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}   $*"; }
error() { echo -e "${RED}[error]${NC}  $*" >&2; exit 1; }

# ── Python ──────────────────────────────────────────────────────────────────
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c 'import sys; print(sys.version_info[:2])')
        if "$cmd" -c 'import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)' 2>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    fi
done
[[ -z "$PYTHON" ]] && error "Python 3.12+ is required but was not found."
info "Using $($PYTHON --version)"

PIP="$PYTHON -m pip"
REQUIREMENTS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/requirements.txt"

# ── Packages ─────────────────────────────────────────────────────────────────
# requirements.txt is the one list of versions (the README and CI read it too).
info "Installing packages from requirements.txt (sentence-transformers pulls PyTorch ~1.5 GB on first run)..."
$PIP install --upgrade -r "$REQUIREMENTS"

# ── API key check ────────────────────────────────────────────────────────────
echo ""
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    warn "ANTHROPIC_API_KEY is not set."
    echo "  Add it to your shell profile, e.g.:"
    echo "    echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc"
    echo "  Or place it in ./.env in your project directory."
else
    info "ANTHROPIC_API_KEY is already set."
fi

echo ""
info "Installation complete. Run the tool from your project root:"
echo "  python3 claude_light.py"
