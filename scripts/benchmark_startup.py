#!/usr/bin/env python3
"""
Benchmark script to measure startup time improvements.

This script measures:
1. Directory walk time (rglob optimization)
2. File hashing time (parallel vs sequential)
3. Model loading time (cache-aware loading)
4. Total index time (warm vs cold cache)

Usage:
    python scripts/benchmark_startup.py [--iterations N] [--test-mode small]
"""
import os
import sys
import time
import argparse
import statistics
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

import claude_light.state as state
from claude_light.skeleton import build_skeleton, _get_cached_paths, _invalidate_path_cache, _file_hash, _file_hash_parallel
from claude_light.indexer import index_files
from claude_light.executor import auto_tune, _check_model_cached


def time_func(fn, *args, **kwargs):
    """Time a function call and return (result, elapsed_ms)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed


def benchmark_directory_walk(iterations=3):
    """Benchmark directory walk with caching."""
    print("\n" + "="*60)
    print("BENCHMARK: Directory Walk (rglob)")
    print("="*60)
    
    times = []
    for i in range(iterations):
        _invalidate_path_cache()
        _, elapsed = time_func(_get_cached_paths)
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.1f} ms")
    
    print(f"\n  Mean: {statistics.mean(times):.1f} ms")
    if len(times) > 1:
        print(f"  StdDev: {statistics.stdev(times):.1f} ms")
    return statistics.mean(times)


def benchmark_file_hashing(iterations=3):
    """Benchmark sequential vs parallel file hashing."""
    print("\n" + "="*60)
    print("BENCHMARK: File Hashing (Sequential vs Parallel)")
    print("="*60)
    
    paths = _get_cached_paths()
    source_files = [p for p in paths if p.is_file() and p.suffix.lower() in {
        '.py', '.js', '.ts', '.tsx', '.md', '.txt', '.json', '.yaml', '.yml',
        '.html', '.css', '.scss', '.java', '.go', '.rs', '.cpp', '.c', '.h'
    }]
    
    print(f"  Files to hash: {len(source_files)}")
    
    seq_times = []
    par_times = []
    
    for i in range(iterations):
        # Sequential
        start = time.perf_counter()
        for p in source_files:
            _file_hash(p)
        seq_elapsed = (time.perf_counter() - start) * 1000
        seq_times.append(seq_elapsed)
        
        # Parallel
        _, par_elapsed = time_func(_file_hash_parallel, source_files)
        par_times.append(par_elapsed)
        
        speedup = seq_elapsed / par_elapsed if par_elapsed > 0 else float('inf')
        print(f"  Iteration {i+1}: Sequential={seq_elapsed:.1f} ms, Parallel={par_elapsed:.1f} ms ({speedup:.2f}x faster)")
    
    seq_mean = statistics.mean(seq_times)
    par_mean = statistics.mean(par_times)
    speedup = seq_mean / par_mean if par_mean > 0 else float('inf')
    
    print(f"\n  Sequential Mean: {seq_mean:.1f} ms")
    print(f"  Parallel Mean:   {par_mean:.1f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    return seq_mean, par_mean


def benchmark_model_loading(iterations=1):
    """Benchmark model loading with cache check."""
    print("\n" + "="*60)
    print("BENCHMARK: Model Loading")
    print("="*60)
    
    # Test with different file counts to trigger different models
    test_cases = [
        (30, "all-MiniLM-L6-v2", "small repo (<50 files)"),
        (100, "all-mpnet-base-v2", "medium repo (50-200 files)"),
        (300, "nomic-ai/nomic-embed-text-v1.5", "large repo (>200 files)"),
    ]
    
    for n_files, model_name, description in test_cases:
        print(f"\n  Scenario: {description}")
        
        # Check if model is cached
        is_cached = _check_model_cached(model_name)
        print(f"    Model cached: {is_cached}")
        
        if not is_cached:
            print(f"    (Skipping load time - model not cached locally)")
            continue
        
        # Time model load with load_model=True
        state.embedder = None
        state.EMBED_MODEL = None
        
        start = time.perf_counter()
        auto_tune([Path(f"fake_file_{i}.py") for i in range(n_files)], quiet=True, load_model=True)
        load_elapsed = (time.perf_counter() - start) * 1000
        
        print(f"    Load time: {load_elapsed:.1f} ms")
        
        # Time model check with load_model=False (warm cache scenario)
        start = time.perf_counter()
        auto_tune([Path(f"fake_file_{i}.py") for i in range(n_files)], quiet=True, load_model=False)
        check_elapsed = (time.perf_counter() - start) * 1000
        
        print(f"    Check-only time: {check_elapsed:.1f} ms")
        print(f"    Savings: {load_elapsed - check_elapsed:.1f} ms")
    
    return None


def benchmark_full_startup(iterations=3):
    """Benchmark full startup time (skeleton + index)."""
    print("\n" + "="*60)
    print("BENCHMARK: Full Startup (Skeleton + Index)")
    print("="*60)
    
    times = []
    for i in range(iterations):
        # Reset state
        _invalidate_path_cache()
        state.chunk_store.clear()
        state._file_hashes.clear()
        state._skeleton_md_hashes.clear()
        state._skeleton_md_parts.clear()
        state.embedder = None
        state.EMBED_MODEL = None
        
        # Time full startup
        start = time.perf_counter()
        build_skeleton()
        index_files(quiet=True)
        elapsed = (time.perf_counter() - start) * 1000
        
        times.append(elapsed)
        cache_hit = len(state._file_hashes) - len([f for f in state._source_files if str(f) not in state._file_hashes])
        print(f"  Iteration {i+1}: {elapsed:.1f} ms (cache hits: {cache_hit}/{len(state._source_files)})")
    
    print(f"\n  Mean: {statistics.mean(times):.1f} ms")
    if len(times) > 1:
        print(f"  StdDev: {statistics.stdev(times):.1f} ms")
    
    return statistics.mean(times)


def main():
    parser = argparse.ArgumentParser(description="Benchmark claude_light startup performance")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for averaging")
    parser.add_argument("--test-mode", choices=["small", "medium", "large", "extra-large"],
                        help="Run in test mode with a synthetic codebase")
    args = parser.parse_args()
    
    print("Claude Light Startup Benchmark")
    print("="*60)
    
    if args.test_mode:
        from tests.utilities.test_mocks import MockManager
        manager = MockManager(args.test_mode)
        manager.start()
        print(f"Running in test mode: {args.test_mode}")
    
    # Run benchmarks
    walk_time = benchmark_directory_walk(args.iterations)
    seq_hash, par_hash = benchmark_file_hashing(args.iterations)
    benchmark_model_loading(1)
    full_startup = benchmark_full_startup(args.iterations)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Directory Walk:     {walk_time:.1f} ms")
    print(f"  File Hashing:       {par_hash:.1f} ms (parallel, {seq_hash/par_hash:.2f}x faster than sequential)")
    print(f"  Full Startup:       {full_startup:.1f} ms")
    print("\nOptimizations applied:")
    print("  ✓ Single directory walk (cached across skeleton + indexer)")
    print("  ✓ Parallel file hashing (ThreadPoolExecutor)")
    print("  ✓ Cache-aware model loading (skip on warm cache)")
    
    if args.test_mode:
        manager.stop()


if __name__ == "__main__":
    main()
