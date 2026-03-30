#!/usr/bin/env bash
# Generate CycloneDX SBOM for claude-light
# Requires: pip install cyclonedx-bom

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "[sbom] Generating CycloneDX SBOM..."

if ! command -v cyclonedx-py &> /dev/null; then
    echo "[sbom] cyclonedx-py not found. Installing..."
    pip install cyclonedx-bom --quiet
fi

cd "$PROJECT_ROOT"
cyclonedx-py requirements -i requirements.txt -o sbom.json

echo "[sbom] SBOM generated at $(pwd)/sbom.json"
