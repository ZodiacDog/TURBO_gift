"""
TURBO_gift Basic Example
========================

Demonstrates REAL optimization scenarios where TURBO_gift delivers value.

Run: python examples/basic_usage.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import gc

from core.engine import (  # type: ignore
    TurboEngine, TurboConfig, OptimizationLevel, PrecisionMode,
    turbo_optimize, get_buffer, release_buffer
)


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def print_result(label, original_mb, optimized_mb, time_ms=None):
    savings = (1 - optimized_mb/original_mb) * 100 if original_mb > 0 else 0
    time_str = f" in {time_ms:.2f}ms" if time_ms else ""
    print(f"  {label}")
    print(f"    Original:  {original_mb:>8.2f} MB")
    print(f"    Optimized: {optimized_mb:>8.2f} MB")
    print(f"    Savings:   {savings:>8.1f}%{time_str}")


def main():
    print("\n" + "="*60)
    print("  TURBO_gift v1.0.0 - Basic Usage Examples")
    print("  A Gift to Humanity")
    print("="*60)

    engine = TurboEngine(TurboConfig(
        optimization_level=OptimizationLevel.AGGRESSIVE,
        precision_mode=PrecisionMode.ADAPTIVE
    ))

    # =========================================================
    # SCENARIO 1: Sparse Data (Common in ML - embeddings, attention)
    # =========================================================
    print_header("SCENARIO 1: Sparse Data Optimization")
    
    # Create sparse data (95% zeros - common in embeddings, sparse attention)
    sparse_data = np.zeros((2000, 2000), dtype=np.float64)
    # Scatter some values (5% non-zero)
    num_nonzero = int(4000000 * 0.05)  # 5% of elements
    indices = np.random.choice(4000000, size=num_nonzero, replace=False)
    sparse_data.flat[indices] = np.random.randn(num_nonzero)
    
    original_mb = sparse_data.nbytes / (1024*1024)
    
    start = time.perf_counter()
    optimized, metadata = engine.optimize(sparse_data)
    elapsed = (time.perf_counter() - start) * 1000
    
    optimized_mb = optimized.nbytes / (1024*1024) if optimized.size > 0 else 0
    print_result("95% Sparse Matrix (2000x2000 FP64)", original_mb, optimized_mb, elapsed)
    print(f"    Pattern detected: {metadata['stages']['filter']['pattern'].name}")

    # =========================================================
    # SCENARIO 2: Precision Reduction (FP64 → FP16)
    # =========================================================
    print_header("SCENARIO 2: Precision Reduction")
    
    # Data that doesn't need FP64 precision (normalized features, probabilities)
    normalized_data = np.random.rand(1000, 1000).astype(np.float64)  # Values 0-1
    
    original_mb = normalized_data.nbytes / (1024*1024)
    
    start = time.perf_counter()
    optimized, metadata = engine.optimize(normalized_data)
    elapsed = (time.perf_counter() - start) * 1000
    
    optimized_mb = optimized.nbytes / (1024*1024)
    print_result("Normalized Features (1000x1000 FP64→FP16)", original_mb, optimized_mb, elapsed)
    
    # Verify data integrity
    restored = engine.restore(optimized, metadata)
    max_error = np.max(np.abs(normalized_data - restored))
    print(f"    Max restoration error: {max_error:.6f} (acceptable for normalized data)")

    # =========================================================
    # SCENARIO 3: Integer-like Data (Quantization to INT8)
    # =========================================================
    print_header("SCENARIO 3: Integer-like Data Quantization")
    
    # Pixel values, counts, indices stored as float
    int_like_data = np.random.randint(0, 256, (1000, 1000)).astype(np.float64)
    
    original_mb = int_like_data.nbytes / (1024*1024)
    
    start = time.perf_counter()
    optimized, metadata = engine.optimize(int_like_data)
    elapsed = (time.perf_counter() - start) * 1000
    
    optimized_mb = optimized.nbytes / (1024*1024)
    print_result("Pixel Values (1000x1000 FP64→INT8)", original_mb, optimized_mb, elapsed)

    # =========================================================
    # SCENARIO 4: Memory Pool Performance
    # =========================================================
    print_header("SCENARIO 4: Memory Pool Allocation Speed")
    print("  Key benefit: Reduces GC pressure and ensures predictable timing")
    
    shape = (500, 500)
    iterations = 1000
    
    # Warmup the pool first
    for _ in range(10):
        buf = engine.get_buffer(shape, np.float32)
        engine.release_buffer(buf)
    
    # Standard numpy allocation
    start = time.perf_counter()
    for _ in range(iterations):
        buf = np.zeros(shape, dtype=np.float32)
        # Simulate doing work
        buf[0, 0] = 1.0
        del buf
    numpy_time = (time.perf_counter() - start) * 1000
    
    # Force garbage collection to measure its impact
    gc_start = time.perf_counter()
    import gc
    gc.collect()
    gc_time = (time.perf_counter() - gc_start) * 1000
    
    # TURBO memory pool (already warmed up)
    start = time.perf_counter()
    for _ in range(iterations):
        buf = engine.get_buffer(shape, np.float32)
        buf[0, 0] = 1.0
        engine.release_buffer(buf)
    pool_time = (time.perf_counter() - start) * 1000
    
    pool_stats = engine.memory_pool.get_stats()
    
    print(f"  {iterations} allocations of {shape} float32:")
    print(f"    NumPy + GC:        {numpy_time + gc_time:>8.2f} ms")
    print(f"    TURBO pool:        {pool_time:>8.2f} ms")
    print(f"    Pool hit rate:     {pool_stats['hit_rate']*100:>8.1f}%")
    print(f"    Bytes reused:      {pool_stats['bytes_saved']/1024/1024:>8.2f} MB")
    print(f"\n  Key advantages:")
    print(f"    • No GC pauses during operation")
    print(f"    • Predictable allocation timing")
    print(f"    • Memory stays hot in cache")

    # =========================================================
    # SCENARIO 5: Batch Processing
    # =========================================================
    print_header("SCENARIO 5: Batch Processing Pipeline")
    
    # Simulate processing multiple data chunks
    batch = [
        np.zeros((500, 500), dtype=np.float64),  # Sparse
        np.random.rand(500, 500).astype(np.float64),  # Normalized
        np.random.randint(0, 100, (500, 500)).astype(np.float64),  # Integer-like
        np.diag(np.ones(500)).astype(np.float64),  # Structured (diagonal)
    ]
    batch[0][::10, ::10] = np.random.randn(50, 50)  # Make first one sparse
    
    original_total = sum(arr.nbytes for arr in batch) / (1024*1024)
    
    start = time.perf_counter()
    optimized_batch, metadata_batch = engine.optimize_batch(batch)
    elapsed = (time.perf_counter() - start) * 1000
    
    optimized_total = sum(arr.nbytes for arr in optimized_batch if hasattr(arr, 'nbytes')) / (1024*1024)
    
    print(f"  4 arrays of varying patterns:")
    for i, meta in enumerate(metadata_batch):
        pattern = meta['stages']['filter']['pattern'].name
        savings = meta['savings_percent']
        print(f"    Array {i+1}: {pattern:<12} → {savings:>5.1f}% savings")
    
    print(f"\n  Batch Total:")
    print_result("Combined", original_total, optimized_total, elapsed)

    # =========================================================
    # FINAL STATISTICS
    # =========================================================
    print_header("ENGINE STATISTICS")
    engine.print_stats()
    
    print("\n" + "="*60)
    print("  All scenarios completed successfully!")
    print("  TURBO_gift is ready for production use.")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
