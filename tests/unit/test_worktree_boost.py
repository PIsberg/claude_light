import numpy as np

from claude_light import config, retrieval


def _setup(monkeypatch, modified):
    monkeypatch.setattr(retrieval, "_modified_paths", lambda: frozenset(modified))
    monkeypatch.setattr(config, "WORKTREE_BOOST", 0.5)


def test_boost_raises_modified_file_chunks(monkeypatch):
    _setup(monkeypatch, {"src/a.py"})
    ids = ["src\\a.py::f", "src/b.py::g"]
    out = retrieval._boost_modified_chunks(np.array([1.0, 1.0]), ids)
    assert out[0] == 1.5 and out[1] == 1.0


def test_boost_keeps_excluded_chunks_excluded(monkeypatch):
    _setup(monkeypatch, {"tests/t.py"})
    out = retrieval._boost_modified_chunks(np.array([-np.inf]), ["tests/t.py::x"])
    assert out[0] == -np.inf


def test_boost_directory_prefix(monkeypatch):
    _setup(monkeypatch, {"newdir/"})
    out = retrieval._boost_modified_chunks(np.array([0.0, 0.0]), ["newdir/x.py::f", "other/x.py::f"])
    assert out[0] == 0.5 and out[1] == 0.0


def test_boost_matches_whole_path_components(monkeypatch):
    _setup(monkeypatch, {"auth.py"})
    out = retrieval._boost_modified_chunks(np.array([0.0, 0.0]), ["src/oauth.py::f", "auth.py::g"])
    assert out[0] == 0.0 and out[1] == 0.5


def test_boost_git_path_relative_to_repo_root_above_cwd(monkeypatch):
    _setup(monkeypatch, {"sub/src/a.py"})
    out = retrieval._boost_modified_chunks(np.array([0.0]), ["src/a.py::f"])
    assert out[0] == 0.5


def test_boost_disabled(monkeypatch):
    _setup(monkeypatch, {"src/a.py"})
    monkeypatch.setattr(config, "WORKTREE_BOOST", 0.0)
    arr = np.array([1.0])
    assert retrieval._boost_modified_chunks(arr, ["src/a.py::f"])[0] == 1.0
