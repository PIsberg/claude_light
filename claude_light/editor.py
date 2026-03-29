import sys
import re
import difflib
from pathlib import Path

from claude_light.ui import _T_EDIT, _T_ERR, _ANSI_CYAN, _ANSI_GREEN, _ANSI_RED, _ANSI_BOLD, _ANSI_RESET, show_diff
from claude_light.linter import _lint_content
from claude_light import git_manager

_ANY_BLOCK = re.compile(r"```[a-zA-Z0-9_+-]*(?::([^\n`]+))?[ \t]*\n(.*?)```", re.DOTALL)
_SR_PATTERN = re.compile(
    r"^<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE\n?$",
    re.DOTALL,
)

def parse_edit_blocks(text):
    edits = []
    for m in _ANY_BLOCK.finditer(text):
        raw_path = m.group(1)
        path = raw_path.strip() if raw_path else ""
        body = m.group(2)
        
        if not path:
            lines = body.lstrip().splitlines()
            for line in lines[:3]:
                import re
                mt = re.match(r"^(?:#|//|/\*+|<!--)\s*([\w./\\-]+\.\w+)\s*(?:\*/|-->)?\s*$", line.strip())
                if mt:
                    path = mt.group(1)
                    break
                    
        if not path:
            continue
            
        path = path.lstrip("/\\")
        while path.startswith("./") or path.startswith(".\\"):
            path = path[2:]
            
        sr = _SR_PATTERN.match(body)
        if sr:
            edits.append({"type": "edit", "path": path,
                          "search": sr.group(1), "replace": sr.group(2)})
        else:
            edits.append({"type": "new", "path": path, "content": body})
    return edits

def _resolve_new_content(edit):
    p = Path(edit["path"])
    if edit["type"] == "new":
        old = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
        return old, edit["content"]

    if not p.exists():
        raise FileNotFoundError(f"File not found: {edit['path']}")
    
    old = p.read_text(encoding="utf-8", errors="ignore")
    search = edit["search"]
    replace = edit["replace"]
    
    if search in old:
        return old, old.replace(search, replace, 1)
        
    s_strip = search.strip("\r\n")
    r_strip = replace.strip("\r\n")
    if s_strip and s_strip in old:
        return old, old.replace(s_strip, r_strip, 1)
        
    norm_old = old.replace("\r\n", "\n")
    norm_search = search.replace("\r\n", "\n")
    norm_replace = replace.replace("\r\n", "\n")
    
    if norm_search in norm_old:
        return old, norm_old.replace(norm_search, norm_replace, 1)
        
    norm_s_strip = norm_search.strip("\n")
    norm_r_strip = norm_replace.strip("\n")
    if norm_s_strip and norm_s_strip in norm_old:
        return old, norm_old.replace(norm_s_strip, norm_r_strip, 1)
        
    s_lines = norm_search.splitlines()
    o_lines = norm_old.splitlines()
    
    while s_lines and not s_lines[0].strip(): s_lines.pop(0)
    while s_lines and not s_lines[-1].strip(): s_lines.pop()
    
    if s_lines:
        s_stripped = [l.strip() for l in s_lines]
        o_stripped = [l.strip() for l in o_lines]
        
        slen = len(s_stripped)
        for i in range(len(o_stripped) - slen + 1):
            if o_stripped[i:i+slen] == s_stripped:
                orig_lead = len(o_lines[i]) - len(o_lines[i].lstrip(" \t"))
                search_lead = len(s_lines[0]) - len(s_lines[0].lstrip(" \t"))
                
                new_r_lines = []
                for rl in norm_replace.splitlines():
                    r_trim = rl.lstrip(" \t")
                    if not r_trim:
                        new_r_lines.append("")
                        continue
                    rel_indent = max(0, len(rl) - len(r_trim) - search_lead)
                    new_r_lines.append(" " * (orig_lead + rel_indent) + r_trim)
                
                prefix = "\n".join(o_lines[:i])
                suffix = "\n".join(o_lines[i+slen:])
                
                res = []
                if prefix: res.append(prefix)
                res.append("\n".join(new_r_lines))
                if suffix: res.append(suffix)
                
                final_str = "\n".join(res)
                if norm_old.endswith("\n") and not final_str.endswith("\n"):
                    final_str += "\n"
                return old, final_str
                
    if s_lines:
        s_text = "\n".join(s_stripped)
        best_ratio, best_idx, best_len = 0, -1, 0
        
        start_search = max(1, slen - 2)
        end_search = min(len(o_stripped), slen + 2)
        
        for w_size in range(start_search, end_search + 1):
            for i in range(len(o_stripped) - w_size + 1):
                window = "\n".join(o_stripped[i:i+w_size])
                ratio = difflib.SequenceMatcher(None, s_text, window).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = i
                    best_len = w_size
                    
        threshold = 0.70 if slen <= 3 else 0.82
        if best_ratio > threshold:
            orig_lead = len(o_lines[best_idx]) - len(o_lines[best_idx].lstrip(" \t"))
            search_lead = len(s_lines[0]) - len(s_lines[0].lstrip(" \t"))
            
            new_r_lines = []
            for rl in norm_replace.splitlines():
                r_trim = rl.lstrip(" \t")
                if not r_trim:
                    new_r_lines.append("")
                    continue
                rel_indent = max(0, len(rl) - len(r_trim) - search_lead)
                new_r_lines.append(" " * (orig_lead + rel_indent) + r_trim)
                
            prefix = "\n".join(o_lines[:best_idx])
            suffix = "\n".join(o_lines[best_idx+best_len:])
            
            res = []
            if prefix: res.append(prefix)
            res.append("\n".join(new_r_lines))
            if suffix: res.append(suffix)
            
            final_str = "\n".join(res)
            if norm_old.endswith("\n") and not final_str.endswith("\n"):
                final_str += "\n"
            return old, final_str

    raise ValueError(f"SEARCH block not found in {edit['path']} (even with fuzzy sequence matching)")


def apply_edits(edits, check_only=False, explanation="", auto_apply=False):
    """
    Apply file edits from Claude's response.
    
    Args:
        edits: List of edit dictionaries (from parse_edit_blocks).
        check_only: If True, only check edits for syntax errors, don't write files.
        explanation: Brief explanation of changes (used in auto-commit message).
        
    Returns:
        If check_only=True, returns list of lint errors (empty if no errors).
        If check_only=False, returns None after applying changes (or early on error).
    """
    if not edits:
        if not check_only:
            print(f"{_T_EDIT} No file blocks found in response.")
        return [] if check_only else None

    resolved = []
    lint_errors = []
    for e in edits:
        try:
            old, new = _resolve_new_content(e)
            if check_only:
                err = _lint_content(e["path"], new)
                if err:
                    lint_errors.append(f"SyntaxError in {e['path']}:\n{err}")
            resolved.append({**e, "_old": old, "_new": new})
        except Exception as ex:
            if check_only:
                lint_errors.append(f"Edit failed for {e['path']}: {ex}")
            else:
                print(f"{_T_ERR} {ex}")
            resolved.append({**e, "_skip": True})

    if check_only:
        return lint_errors

    applicable = [r for r in resolved if not r.get("_skip")]
    if not applicable:
        print(f"{_T_EDIT} No applicable changes.\n")
        return None

    print(f"\n{_ANSI_BOLD}{'═'*56}{_ANSI_RESET}")
    for r in applicable:
        is_new = r["type"] == "new" and not Path(r["path"]).exists()
        tag = f"{_ANSI_GREEN}NEW{_ANSI_RESET}" if is_new else f"{_ANSI_CYAN}EDIT{_ANSI_RESET}"
        print(f"  [{tag}] {r['path']}")
    print(f"{_ANSI_BOLD}{'═'*56}{_ANSI_RESET}\n")

    for r in applicable:
        path = r["path"]
        print(f"{_ANSI_CYAN}── {path} {_ANSI_RESET}" + "─" * max(0, 50 - len(path)))
        show_diff(path, r["_new"], old_content=r["_old"])

    if auto_apply:
        auto = True
    elif not sys.stdin.isatty():
        auto = True
    else:
        print(f"Apply {_ANSI_BOLD}{len(applicable)}{_ANSI_RESET} change(s)? "
              f"[{_ANSI_GREEN}y{_ANSI_RESET}/{_ANSI_RED}n{_ANSI_RESET}] ", end="", flush=True)
        try:
            auto = input().strip().lower() in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            auto = False

    if not auto:
        print(f"{_T_EDIT} Cancelled.\n")
        return

    written = 0
    written_files = []
    for r in applicable:
        try:
            p = Path(r["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(r["_new"], encoding="utf-8")
            print(f"{_T_EDIT} Wrote {_ANSI_CYAN}{r['path']}{_ANSI_RESET}")
            written += 1
            written_files.append(r["path"])
        except Exception as ex:
            print(f"{_T_ERR} Failed to write {r['path']}: {ex}")

    print(f"{_T_EDIT} {_ANSI_GREEN}{written}/{len(applicable)}{_ANSI_RESET} change(s) applied.\n")
    
    # Attempt auto-commit if in a git repo
    if written_files and git_manager.is_git_repo():
        git_manager.auto_commit(written_files, explanation)
