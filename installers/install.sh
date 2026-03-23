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
        if "$cmd" -c 'import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)' 2>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    fi
done
[[ -z "$PYTHON" ]] && error "Python 3.9+ is required but was not found."
info "Using $($PYTHON --version)"

PIP="$PYTHON -m pip"

# ── Required packages ────────────────────────────────────────────────────────
info "Installing required packages (sentence-transformers pulls PyTorch ~1.5 GB on first run)..."
$PIP install --upgrade sentence-transformers numpy watchdog anthropic prompt_toolkit

# ── Optional packages ────────────────────────────────────────────────────────
info "Installing optional packages (tree-sitter, rich, einops)..."
$PIP install --upgrade \
    tree-sitter \
    tree-sitter-java \
    tree-sitter-python \
    tree-sitter-go \
    tree-sitter-rust \
    tree-sitter-javascript \
    tree-sitter-typescript \
    rich \
    einops \
    || warn "Some optional packages failed to install — the tool will still work, with reduced functionality."

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
