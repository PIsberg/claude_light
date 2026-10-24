"""
Tests for the hybrid index (#61): a static model indexes first so retrieval
works within seconds, a dense model fills in afterwards, and retrieval fuses
the two.

The models are faked with a deterministic bag-of-words embedder, so the tests
need no download and similarity still means something.
"""

import os
import re
import sys
import tempfile
import threading
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import claude_light.state as state
from claude_light import config, indexer, retrieval


class _BagOfWords:
    """Stands in for a SentenceTransformer: hashed word counts, normalised."""

    def __init__(self, name, dims=256):
        self.name = name
        self.dims = dims
        self.calls = 0

    def encode(self, texts, normalize_embeddings=True, **_kw):
        single = isinstance(texts, str)
        rows = []
        for text in [texts] if single else texts:
            self.calls += 1
            v = np.zeros(self.dims, dtype=np.float32)
            for word in re.findall(r"[a-z_]+", text.lower()):
                v[zlib.crc32(f"{self.name}:{word}".encode()) % self.dims] += 1.0
            n = np.linalg.norm(v)
            rows.append(v / n if n else v)
        return rows[0] if single else np.array(rows)


FILES = {
    "billing.py": "def calculate_invoice_total(items, tax):\n    return sum(items) * (1 + tax)\n",
    "auth.py": "def login_user(name, password):\n    return check_password(name, password)\n",
    "report.py": "def render_sales_report(rows):\n    return '\\n'.join(rows)\n",
}


class _HybridProject(unittest.TestCase):
    """A temp project with the hybrid index switched on and fake models."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.orig_cwd = os.getcwd()
        os.chdir(self.tmp)
        for name, text in FILES.items():
            Path(name).write_text(text, encoding="utf-8")
        self.models = {}
        # index_files rebinds the hash dicts; later tests hold the originals.
        self._saved = {k: getattr(state, k) for k in (
            "fast_embedder", "embedder", "EMBED_MODEL", "TOP_K", "last_retrieval_mode",
            "_file_hashes", "_file_stats", "_source_files")}
        self._store = dict(state.chunk_store)
        state.chunk_store.clear()
        state.fast_embedder = state.embedder = None
        state.embedder_ready.clear()
        state.dense_ready.clear()
        for p in (patch.object(config, "HYBRID_RETRIEVAL", True),
                  patch.object(indexer, "_load_embedding_model", side_effect=self._load),
                  patch("claude_light.executor._load_embedding_model", side_effect=self._load)):
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        os.chdir(self.orig_cwd)
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)
        for k, v in self._saved.items():
            setattr(state, k, v)
        state.chunk_store.clear()
        state.chunk_store.update(self._store)
        state.embedder_ready.clear()
        state.dense_ready.clear()

    def _load(self, name, quiet=False):
        self.models.setdefault(name, _BagOfWords(name))
        return self.models[name]

    def _files_retrieved(self, question):
        _ctx, hits = retrieval.retrieve(question, token_budget=3000, effort="medium")
        return [h[0].split("::")[0] for h in hits]


class TestHybridScores(unittest.TestCase):

    def test_fuses_z_scores_so_neither_scale_dominates(self):
        print("\n  ▶ TestHybridScores.test_fuses_z_scores_so_neither_scale_dominates")
        fast = np.array([0.30, 0.10, 0.10, 0.10])     # static: chunk 0
        dense = np.array([0.10, 0.90, 0.10, 0.10])    # dense: chunk 1, bigger raw scale
        fused = retrieval.hybrid_scores(fast, dense, fast_min=0.2, dense_min=0.2)
        self.assertAlmostEqual(fused[0], fused[1])
        self.assertGreater(fused[1], fused[2])

    def test_nothing_related_returns_none(self):
        print("\n  ▶ TestHybridScores.test_nothing_related_returns_none")
        weak = np.array([0.05, 0.10, 0.12])
        self.assertIsNone(retrieval.hybrid_scores(weak, weak, fast_min=0.22, dense_min=0.21))
        self.assertIsNotNone(retrieval.hybrid_scores(weak, weak + 0.2, fast_min=0.22, dense_min=0.21))

    def test_static_alone_until_dense_scores_exist(self):
        print("\n  ▶ TestHybridScores.test_static_alone_until_dense_scores_exist")
        fast = np.array([0.5, 0.1, 0.3])
        fused = retrieval.hybrid_scores(fast, None, fast_min=0.2)
        self.assertEqual(int(np.argmax(fused)), 0)


class TestHybridIndex(_HybridProject):

    def test_static_index_answers_before_the_dense_vectors_exist(self):
        print("\n  ▶ TestHybridIndex.test_static_index_answers_before_the_dense_vectors_exist")
        indexer.index_files(quiet=True, dense="skip")
        self.assertTrue(state.chunk_store)
        self.assertTrue(all("emb_fast" in e and "emb" not in e for e in state.chunk_store.values()))
        self.assertNotIn(config.DENSE_EMBED_MODEL, self.models)  # never loaded
        self.assertEqual(self._files_retrieved("calculate invoice total with tax")[0], "billing.py")
        self.assertIn("static", state.last_retrieval_mode)

    def test_dense_pass_fills_every_chunk_and_retrieval_fuses(self):
        print("\n  ▶ TestHybridIndex.test_dense_pass_fills_every_chunk_and_retrieval_fuses")
        indexer.index_files(quiet=True, dense="sync")
        self.assertTrue(all("emb" in e for e in state.chunk_store.values()))
        self.assertTrue(state.dense_ready.is_set())
        self.assertEqual(self._files_retrieved("login user password check")[0], "auth.py")
        self.assertIn("hybrid", state.last_retrieval_mode)

    def test_background_pass_does_not_block_index_files(self):
        print("\n  ▶ TestHybridIndex.test_background_pass_does_not_block_index_files")
        gate = threading.Event()
        dense = _BagOfWords(config.DENSE_EMBED_MODEL)
        real_encode = dense.encode
        dense.encode = lambda *a, **k: (gate.wait(5), real_encode(*a, **k))[1]
        self.models[config.DENSE_EMBED_MODEL] = dense
        indexer.index_files(quiet=True, dense="background")
        self.assertFalse(state.dense_ready.is_set())   # still embedding
        self.assertTrue(self._files_retrieved("render sales report rows"))
        gate.set()
        self.assertTrue(state.dense_ready.wait(5))

    def test_unrelated_question_retrieves_nothing(self):
        print("\n  ▶ TestHybridIndex.test_unrelated_question_retrieves_nothing")
        indexer.index_files(quiet=True, dense="sync")
        self.assertEqual(self._files_retrieved("bake sourdough bread"), [])

    def test_cache_reuses_vectors_and_drops_only_a_changed_model(self):
        print("\n  ▶ TestHybridIndex.test_cache_reuses_vectors_and_drops_only_a_changed_model")
        indexer.index_files(quiet=True, dense="sync")
        state.chunk_store.clear()
        state.fast_embedder = state.embedder = None
        self.models.clear()
        indexer.index_files(quiet=True, dense="skip")
        self.assertEqual(self.models[config.FAST_EMBED_MODEL].calls, 0)  # all from the cache
        self.assertTrue(all("emb" in e for e in state.chunk_store.values()))

        state.chunk_store.clear()
        state.fast_embedder = state.embedder = None
        with patch.object(config, "DENSE_EMBED_MODEL", "some-other-dense-model"):
            indexer.index_files(quiet=True, dense="skip")
        entries = list(state.chunk_store.values())
        self.assertTrue(all("emb_fast" in e for e in entries))
        self.assertFalse(any("emb" in e for e in entries))  # re-embedded later, not reused

    def test_saved_file_gets_both_vectors(self):
        print("\n  ▶ TestHybridIndex.test_saved_file_gets_both_vectors")
        indexer.index_files(quiet=True, dense="sync")
        Path("billing.py").write_text("def refund_order(order_id):\n    return order_id\n", encoding="utf-8")
        with patch("claude_light.indexer.notify"):
            indexer.reindex_file(os.path.join(".", "billing.py"))
        ids = [cid for cid in state.chunk_store if cid.startswith("billing.py")]
        self.assertTrue(ids)
        self.assertTrue(all("emb" in state.chunk_store[c] and "emb_fast" in state.chunk_store[c] for c in ids))
        self.assertEqual(self._files_retrieved("where is refund_order with order_id")[0], "billing.py")


if __name__ == "__main__":
    unittest.main()
