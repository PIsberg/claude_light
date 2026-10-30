"""Fail a PR that changes a PlantUML source without its rendered PNG (#107).

indexing_flow.puml was edited while Claude_Light_Indexing_Flow.png stayed at a
render months older (#67). A fresh render is not compared byte for byte: PNG
output depends on the PlantUML version, Graphviz and fonts. This only checks
that a changed .puml under docs/plantuml/ arrives with its PNG, which is named
by the diagram's @startuml line.

    python tests/linting/check_diagram_pngs.py origin/main
"""

import re
import subprocess
import sys
from pathlib import Path

DIAGRAM_DIR = "docs/plantuml/"
_STARTUML = re.compile(r"^@startuml\s+(\S+)", re.M)


def png_for(puml: Path) -> str:
    match = _STARTUML.search(puml.read_text(encoding="utf-8"))
    return f"{match.group(1) if match else puml.stem}.png"


def stale_diagrams(changed: list[str], root: Path) -> list[tuple[str, str]]:
    """(source, png) for each changed source whose PNG did not change."""
    changed_set = set(changed)
    stale = []
    for path in changed:
        if not (path.startswith(DIAGRAM_DIR) and path.endswith(".puml")):
            continue
        puml = root / path
        if not puml.is_file():  # deleted in this change
            continue
        png = f"{DIAGRAM_DIR}{png_for(puml)}"
        if png not in changed_set:
            stale.append((path, png))
    return stale


def main(argv: list[str]) -> int:
    base = argv[1] if len(argv) > 1 else "origin/main"
    changed = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    stale = stale_diagrams(changed, Path.cwd())
    for puml, png in stale:
        print(f"ERROR: {puml} changed but {png} did not. Render it with "
              f"`java -jar docs/plantuml/plantuml.jar {puml}` and commit the PNG.")
    if not stale:
        print("Diagram PNGs OK.")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
