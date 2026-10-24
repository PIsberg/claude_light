#!/usr/bin/env python3
"""
Embedding-model benchmark on this repository (issue #61).

For every question in tests/fixtures/repo_questions.json (a question about this
repo and the file that answers it, or an unrelated question that should get no
context), this runs claude_light's real retrieval selection and reports:

  * file-level MRR: rank of the answering file by its best chunk
  * selected: did the answering file reach the context the model would see
    (medium effort budget), and how many files were sent
  * unrelated: how many off-topic questions got no context, as they should

for the single dense models and for the hybrid static + dense fusion.

    python tests/benchmarks/benchmark_models.py                  # default set
    python tests/benchmarks/benchmark_models.py --cache-dir .bench_cache
    python tests/benchmarks/benchmark_models.py --sweep          # tune floors

Embeddings are cached per model in --cache-dir, so re-runs and threshold sweeps
take seconds. Embedding the repo with all-mpnet-base-v2 takes ~6 minutes on CPU.
Run from the repository root. Not collected by pytest.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import claude_light.state as state  # noqa: E402
from claude_light import config, retrieval  # noqa: E402
from claude_light.config import INDEXABLE_EXTENSIONS, _DOC_PREFIX, _QUERY_PREFIX  # noqa: E402
from claude_light.indexer import _is_skipped  # noqa: E402
from claude_light.parsing import chunk_file  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "repo_questions.json"
FAST = config.FAST_EMBED_MODEL
DENSE_MODELS = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
BUDGET = config._RETRIEVAL_BUDGET["medium"]
TEST_POLICY = {"policy": "exclude", "penalty": 0.0}
_TEST_WORDS = re.compile(r"\btests?\b|\btesting\b|\bspec\b|\bpytest\b", re.IGNORECASE)


def apply_test_policy(fused, is_test, question):
    """Adjust fused scores for chunks under tests/, unless the question is about tests."""
    policy = TEST_POLICY["policy"]
    if policy == "keep" or _TEST_WORDS.search(question):
        return fused
    if policy == "exclude":
        return np.where(is_test, -np.inf, fused)
    return fused - TEST_POLICY["penalty"] * is_test


def collect_chunks():
    chunks = []
    for p in sorted(Path(".").rglob("*")):
        if p.is_file() and p.suffix.lower() in INDEXABLE_EXTENSIONS and not _is_skipped(p):
            chunks.extend(chunk_file(str(p), p.read_text(encoding="utf-8", errors="ignore")))
    return chunks


def file_of(chunk_id):
    return chunk_id.split("::")[0].replace("\\", "/")


def embed(model_name, chunks, questions, cache_dir):
    """(chunk matrix, question matrix) for a model, cached on disk by chunk ids."""
    import hashlib
    key = model_name.replace("/", "__")
    ids_path = cache_dir / f"{key}.ids.json"
    # Keyed on the chunk texts and questions, not just the ids: an edited
    # function keeps its id, and its stale vector would be reused silently.
    digest = hashlib.sha256("\0".join([c["text"] for c in chunks] + [q["question"] for q in questions])
                            .encode("utf-8")).hexdigest()
    if ids_path.exists() and json.loads(ids_path.read_text()) == digest:
        return np.load(cache_dir / f"{key}.chunks.npy"), np.load(cache_dir / f"{key}.questions.npy")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")
    print(f"  embedding {len(chunks)} chunks with {model_name}...", flush=True)
    E = model.encode([_DOC_PREFIX.get(model_name, "") + c["text"] for c in chunks],
                     normalize_embeddings=True, batch_size=64)
    Q = model.encode([_QUERY_PREFIX.get(model_name, "") + q["question"] for q in questions],
                     normalize_embeddings=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{key}.chunks.npy", E)
    np.save(cache_dir / f"{key}.questions.npy", Q)
    ids_path.write_text(json.dumps(digest))
    return E, Q


def evaluate(name, ids, files, questions, select, budget=BUDGET):
    """select(i, k) -> list of (chunk_id, score) for question i, or [] for none."""
    k = max(2, round(state.TOP_K * budget / config.TARGET_RETRIEVED_TOKENS))
    out = {"name": name, "identifier": [0, 0], "paraphrase": [0, 0], "unrelated": [0, 0],
           "files_sent": [], "tokens_sent": []}
    for i, q in enumerate(questions):
        picked = select(i, k)
        sent = {file_of(c) for c, _ in picked}
        bucket = out[q["kind"]]
        bucket[1] += 1
        if q["kind"] == "unrelated":
            bucket[0] += not picked
        else:
            bucket[0] += q["file"] in sent
            out["files_sent"].append(len(sent))
        out["tokens_sent"].append(sum(retrieval._estimate_tokens(state.chunk_store[c]["text"])
                                      for c, _ in picked))
    return out


def report(r):
    i, p, u = r["identifier"], r["paraphrase"], r["unrelated"]
    print(f"{r['name']:34} right file sent: identifier {i[0]:2}/{i[1]}  paraphrase {p[0]:2}/{p[1]}"
          f"  | unrelated got nothing {u[0]}/{u[1]}  | files sent median {np.median(r['files_sent']):.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=ROOT / ".claude_light_cache" / "bench")
    ap.add_argument("--sweep", action="store_true", help="also sweep HYBRID_SCORE_GAP and the floors")
    ap.add_argument("--budgets", action="store_true", help="also report recall and tokens per effort budget")
    ap.add_argument("--test-policy", choices=["keep", "exclude", "penalty"], default="exclude",
                    help="for chunks under tests/: keep them, drop them, or subtract --test-penalty "
                         "from their fused score. Not applied when the question mentions tests.")
    ap.add_argument("--test-penalty", type=float, default=1.0, help="fused z-score units (penalty policy)")
    args = ap.parse_args()
    TEST_POLICY["policy"], TEST_POLICY["penalty"] = args.test_policy, args.test_penalty

    questions = json.loads(FIXTURE.read_text(encoding="utf-8"))["questions"]
    chunks = collect_chunks()
    ids = [c["id"] for c in chunks]
    files = [file_of(c) for c in ids]
    is_test = np.array([f.startswith("tests/") for f in files])
    state.chunk_store.clear()
    state.chunk_store.update({c["id"]: {"text": c["text"]} for c in chunks})
    avg_tokens = max(1, sum(len(c["text"]) for c in chunks) // len(chunks) // 4)
    state.TOP_K = max(2, min(15, round(config.TARGET_RETRIEVED_TOKENS / avg_tokens)))
    print(f"{len(chunks)} chunks, {len(set(files))} files, {len(questions)} questions, "
          f"TOP_K {state.TOP_K}, medium budget {BUDGET} tokens\n")

    Ef, Qf = embed(FAST, chunks, questions, args.cache_dir)
    dense = {m: embed(m, chunks, questions, args.cache_dir) for m in DENSE_MODELS}

    def dense_only(model):
        E, Q = dense[model]
        return lambda i, k: retrieval._adaptive_select_pairs(ids, E @ Q[i], BUDGET, k)

    def hybrid(model, gap=None, fast_min=None, dense_min=None, budget=BUDGET):
        def select(i, k):
            s_fast = Ef @ Qf[i]
            s_dense = None
            if model:
                E, Q = dense[model]
                s_dense = E @ Q[i]
            fused = retrieval.hybrid_scores(s_fast, s_dense, fast_min=fast_min,
                                            dense_min=dense_min if dense_min is not None
                                            else config.HYBRID_MIN_SCORE.get(model, config.MIN_SCORE))
            if fused is not None:
                fused = apply_test_policy(fused, is_test, questions[i]["question"])
            return [] if fused is None else retrieval._hybrid_select_pairs(ids, fused, budget, k, gap=gap)
        return select

    print("Dense only (before #61: MIN_SCORE / RELATIVE_SCORE_FLOOR):")
    for m in DENSE_MODELS:
        report(evaluate(m, ids, files, questions, dense_only(m)))
    print("\nHybrid (as configured):")
    report(evaluate("static only (dense pass running)", ids, files, questions, hybrid(None)))
    for m in DENSE_MODELS:
        report(evaluate(f"static + {m}", ids, files, questions, hybrid(m)))

    if args.budgets:
        m = config.DENSE_EMBED_MODEL
        print(f"\nPer-effort budget, static + {m} (tokens = selected chunk text, estimated):")
        for effort, budget in config._RETRIEVAL_BUDGET.items():
            r = evaluate(f"{effort} budget {budget}", ids, files, questions,
                         hybrid(m, budget=budget), budget=budget)
            report(r)
            print(f"{'':34} tokens sent median {np.median(r['tokens_sent']):.0f}"
                  f"  mean {np.mean(r['tokens_sent']):.0f}")

    if args.sweep:
        m = config.DENSE_EMBED_MODEL
        print(f"\nSweep, static + {m}:")
        for gap in (0.75, 1.0, 1.5, 2.0, 3.0):
            report(evaluate(f"gap {gap}", ids, files, questions, hybrid(m, gap=gap)))
        for fmin in (0.18, 0.20, 0.22, 0.24):
            for dmin in (0.17, 0.19, 0.21, 0.23, 0.25):
                report(evaluate(f"fast_min {fmin} dense_min {dmin}", ids, files, questions,
                                hybrid(m, fast_min=fmin, dense_min=dmin)))


if __name__ == "__main__":
    main()
