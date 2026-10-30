"""
Provider isolation (specs/001-gemini-codex-providers: T019).

Covers FR-004, SC-003 and US4 scenario 1. The rule is the one in
contracts/provider.md, "Isolation rule": outside `providers/<name>.py` and the
registry `providers/__init__.py`, no module imports a provider's SDK or holds a
string constant (docstrings excluded) that names its binary.
"""

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
PKG = ROOT / "claude_light"

NAMES = ("claude", "gemini", "codex")
SDKS = {
    "claude": ("anthropic",),
    "gemini": ("google.genai", "google.generativeai"),
    "codex": ("openai",),
}
NAME_RE = re.compile(r"(?<![\w./-])(claude|gemini|codex|agy)(\.exe|\.cmd)?(?![\w.-]| light)")


def _exempt(path: Path, name: str) -> bool:
    rel = path.relative_to(PKG).as_posix()
    return rel in (f"providers/{name}.py", "providers/__init__.py")


def _docstring_nodes(tree):
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                skip.add(id(body[0].value))
    return skip


def _violations():
    found = []
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            for name in NAMES:
                if _exempt(path, name):
                    continue
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    mods = [node.module or ""]
                else:
                    mods = []
                for mod in mods:
                    for sdk in SDKS[name]:
                        if mod == sdk or mod.startswith(sdk + "."):
                            found.append(f"{rel}:{node.lineno}: imports {mod} (the {name} SDK)")
                if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                        and id(node) not in docstrings):
                    for m in NAME_RE.finditer(node.value):
                        owner = "gemini" if m.group(1) == "agy" else m.group(1)
                        if owner == name:
                            found.append(f"{rel}:{node.lineno}: string {node.value!r} "
                                         f"matches {m.group(0)!r} ({name})")
    return found


def test_fr004_sc003_us4s1_no_provider_names_or_sdks_outside_their_modules():
    found = _violations()
    assert not found, "provider isolation broken:\n" + "\n".join(found)


@pytest.mark.parametrize("text", ["claude", "codex.cmd", "claude setup-token",
                                  "gemini", "which codex", "claude.exe", "agy",
                                  'which("agy")'])
def test_fr004_regex_matches_the_binary_forms(text):
    assert NAME_RE.search(text), text


@pytest.mark.parametrize("text", ["claude.md", " claude light", "claude_light",
                                  "CLAUDE.md", "the claude light banner"])
def test_fr004_regex_ignores_file_names_product_name_and_identifiers(text):
    assert not NAME_RE.search(text), text
