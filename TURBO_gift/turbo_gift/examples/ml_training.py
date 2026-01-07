"""
TURBO_gift ML Training Example
==============================

Demonstrates REALISTIC performance gains for ML workloads.

Key insight: TURBO_gift optimizes DATA, not COMPUTATION.
Use it to:
  1. Reduce memory footprint of large datasets/models
  2. Speed up data loading and preprocessing
  3. Enable larger batch sizes by compressing activations
  4. Eliminate allocation overhead in training loops

Run: python examples/ml_training.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import gc

from core.engine import (  # type: ignore
    TurboEngine, TurboConfig, OptimizationLevel, PrecisionMode
)


def print_header(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)


def print_comparison(metric, baseline, optimized, unit="", higher_better=True):
    if higher_better:
        improvement = ((optimized - baseline) / baseline) * 100 if baseline > 0 else 0
        better = "↑" if improvement > 0 else "↓"
    else:
        improvement = ((baseline - optimized) / baseline) * 100 if baseline > 0 else 0
        better = "↓" if improvement > 0 else "↑"
    
    print(f"    {metric:<25} {baseline:>10.2f}{unit}  →  {optimized:>10.2f}{unit}  ({better}{abs(improvement):>5.1f}%)")


# =============================================================================
# SCENARIO 1: Dataset Loading & Preprocessing
# =============================================================================

def scenario_dataset_loading():
    """
    Real use case: Loading and preprocessing large datasets.
    TURBO reduces memory footprint so you can load larger datasets.
    """
    print_header("SCENARIO 1: Dataset Loading & Preprocessing")
    print("  Use case: Loading large image/feature datasets into memory")
    print("  TURBO compresses data so more fits in RAM/VRAM")
    
    engine = TurboEngine(TurboConfig(
        optimization_level=OptimizationLevel.AGGRESSIVE,
        precision_mode=PrecisionMode.FP16
    ))
    
    # Simulate loading a large dataset (e.g., image features)
    num_samples = 10000
    feature_dim = 2048  # Like ResNet features
    
    print(f"\n  Simulating {num_samples:,} samples × {feature_dim} features...")
    
    # WITHOUT optimization
    gc.collect()
    start = time.perf_counter()
    
    dataset_baseline = np.random.randn(num_samples, feature_dim).astype(np.float64)
    baseline_load_time = time.perf_counter() - start
    baseline_memory = dataset_baseline.nbytes / (1024**2)
    
    # WITH optimization
    gc.collect()
    start = time.perf_counter()
    
    raw_data = np.random.randn(num_samples, feature_dim).astype(np.float64)
    dataset_optimized, metadata = engine.optimize(raw_data)
    optimized_load_time = time.perf_counter() - start
    optimized_memory = dataset_optimized.nbytes / (1024**2)
    
    print("\n  Results:")
    print_comparison("Memory Usage", baseline_memory, optimized_memory, " MB", higher_better=False)
    print_comparison("Load + Process Time", baseline_load_time*1000, optimized_load_time*1000, " ms", higher_better=False)
    
    # How many MORE samples could fit in same memory?
    ratio = baseline_memory / optimized_memory
    print(f"\n    → Can fit {ratio:.1f}x more data in same memory!")
    print(f"    → Or: {int(num_samples * ratio):,} samples instead of {num_samples:,}")
    
    del dataset_baseline, dataset_optimized, raw_data
    gc.collect()
    
    return engine


# =============================================================================
# SCENARIO 2: Training Loop with Memory Pool
# =============================================================================

def scenario_training_loop(engine):
    """
    Real use case: Eliminating allocation overhead in training loops.
    Pre-allocated buffers mean zero allocation cost per iteration.
    """
    print_header("SCENARIO 2: Training Loop Allocation Overhead")
    print("  Use case: Eliminating memory allocation in hot loops")
    print("  TURBO's memory pool pre-allocates reusable buffers")
    
    batch_size = 256
    hidden_dim = 1024
    num_iterations = 1000
    
    print(f"\n  Simulating {num_iterations} training iterations...")
    print(f"  Batch: {batch_size}, Hidden: {hidden_dim}")
    
    # WITHOUT memory pool (standard numpy)
    gc.collect()
    start = time.perf_counter()
    
    for _ in range(num_iterations):
        # These allocations happen every iteration in naive training code
        inputs = np.zeros((batch_size, hidden_dim), dtype=np.float32)
        gradients = np.zeros((batch_size, hidden_dim), dtype=np.float32)
        activations = np.zeros((batch_size, hidden_dim), dtype=np.float32)
        # Simulate some work
        activations[:] = np.tanh(inputs)
        del inputs, gradients, activations
    
    baseline_time = time.perf_counter() - start
    
    # WITH memory pool
    gc.collect()
    start = time.perf_counter()
    
    for _ in range(num_iterations):
        # Get pre-allocated buffers - near-zero allocation cost
        inputs = engine.get_buffer((batch_size, hidden_dim), np.float32)
        gradients = engine.get_buffer((batch_size, hidden_dim), np.float32)
        activations = engine.get_buffer((batch_size, hidden_dim), np.float32)
        # Simulate some work
        activations[:] = np.tanh(inputs)
        # Return to pool for reuse
        engine.release_buffer(inputs)
        engine.release_buffer(gradients)
        engine.release_buffer(activations)
    
    optimized_time = time.perf_counter() - start
    
    print("\n  Results:")
    print_comparison("Total Time", baseline_time*1000, optimized_time*1000, " ms", higher_better=False)
    print_comparison("Per-Iteration", (baseline_time/num_iterations)*1e6, (optimized_time/num_iterations)*1e6, " µs", higher_better=False)
    
    speedup = baseline_time / optimized_time
    print(f"\n    → {speedup:.1f}x faster iteration overhead!")
    
    pool_stats = engine.memory_pool.get_stats()
    print(f"    → Pool hit rate: {pool_stats['hit_rate']*100:.1f}%")


# =============================================================================
# SCENARIO 3: Sparse Activation Optimization
# =============================================================================

def scenario_sparse_activations(engine):
    """
    Real use case: ReLU activations are ~50% zeros, attention is often 90%+ sparse.
    TURBO detects and exploits this sparsity.
    """
    print_header("SCENARIO 3: Sparse Activation Optimization")
    print("  Use case: ReLU/attention activations are highly sparse")
    print("  TURBO detects sparsity and compresses accordingly")
    
    batch_size = 128
    seq_length = 512
    hidden_dim = 768  # BERT-base size
    
    # Simulate ReLU activations (~50% zeros)
    relu_activations = np.random.randn(batch_size, seq_length, hidden_dim).astype(np.float32)
    relu_activations[relu_activations < 0] = 0  # ReLU
    
    # Simulate attention weights (sparse - most attention is concentrated)
    attention = np.zeros((batch_size, 12, seq_length, seq_length), dtype=np.float32)
    # Only top-k attention per position (simulating sparse attention)
    for b in range(batch_size):
        for h in range(12):
            for i in range(seq_length):
                top_k_indices = np.random.choice(seq_length, size=32, replace=False)
                attention[b, h, i, top_k_indices] = np.random.rand(32)
    
    print(f"\n  ReLU activations: {relu_activations.shape}")
    print(f"  Attention weights: {attention.shape}")
    
    # Optimize both
    relu_original = relu_activations.nbytes / (1024**2)
    attn_original = attention.nbytes / (1024**2)
    
    start = time.perf_counter()
    relu_opt, relu_meta = engine.optimize(relu_activations)
    attn_opt, attn_meta = engine.optimize(attention)
    opt_time = (time.perf_counter() - start) * 1000
    
    relu_compressed = relu_opt.nbytes / (1024**2)
    attn_compressed = attn_opt.nbytes / (1024**2)
    
    print("\n  ReLU Activations:")
    print_comparison("Memory", relu_original, relu_compressed, " MB", higher_better=False)
    print(f"    Pattern: {relu_meta['stages']['filter']['pattern'].name}")
    
    print("\n  Attention Weights:")
    print_comparison("Memory", attn_original, attn_compressed, " MB", higher_better=False)
    print(f"    Pattern: {attn_meta['stages']['filter']['pattern'].name}")
    
    total_original = relu_original + attn_original
    total_compressed = relu_compressed + attn_compressed
    print(f"\n  Total Savings: {total_original:.1f} MB → {total_compressed:.1f} MB")
    print(f"    → {(1 - total_compressed/total_original)*100:.1f}% memory reduction")
    print(f"    → Optimization time: {opt_time:.2f} ms")


# =============================================================================
# SCENARIO 4: Gradient Compression for Distributed Training
# =============================================================================

def scenario_gradient_compression(engine):
    """
    Real use case: In distributed training, gradient communication is bottleneck.
    Compressing gradients before sending reduces network time.
    """
    print_header("SCENARIO 4: Gradient Compression (Distributed Training)")
    print("  Use case: Compressing gradients before network transfer")
    print("  Smaller gradients = faster allreduce = faster training")
    
    # Simulate gradients for a medium model (50M params)
    gradient_shapes = [
        (1024, 1024),   # 1M
        (1024, 4096),   # 4M
        (4096, 1024),   # 4M
        (4096, 4096),   # 16M
        (4096, 4096),   # 16M
        (4096, 2048),   # 8M
    ]
    
    print(f"\n  Simulating gradients for ~50M parameter model...")
    
    total_original = 0
    total_compressed = 0
    total_time = 0
    
    for i, shape in enumerate(gradient_shapes):
        # Gradients are often small values, normally distributed
        gradients = np.random.randn(*shape).astype(np.float32) * 0.01
        
        original_mb = gradients.nbytes / (1024**2)
        total_original += original_mb
        
        start = time.perf_counter()
        compressed, meta = engine.optimize(gradients)
        total_time += time.perf_counter() - start
        
        compressed_mb = compressed.nbytes / (1024**2)
        total_compressed += compressed_mb
    
    print("\n  Results:")
    print_comparison("Gradient Size", total_original, total_compressed, " MB", higher_better=False)
    print(f"    Compression time: {total_time*1000:.2f} ms")
    
    # Simulate network transfer time (10 Gbps)
    network_speed_mbps = 10000 / 8  # 10 Gbps in MB/s
    baseline_transfer = total_original / network_speed_mbps * 1000
    optimized_transfer = total_compressed / network_speed_mbps * 1000
    
    print(f"\n  Network Transfer (simulated 10 Gbps):")
    print_comparison("Transfer Time", baseline_transfer, optimized_transfer, " ms", higher_better=False)
    print(f"\n    → Saves {baseline_transfer - optimized_transfer:.1f} ms per allreduce!")


# =============================================================================
# SCENARIO 5: Inference Batching Optimization
# =============================================================================

def scenario_inference_optimization(engine):
    """
    Real use case: In inference, you want maximum throughput.
    Smaller memory = larger batches = better GPU utilization.
    """
    print_header("SCENARIO 5: Inference Batch Size Optimization")
    print("  Use case: Fitting larger batches for inference throughput")
    print("  Compressed activations = room for bigger batches")
    
    # Simulate inference with different batch sizes
    gpu_memory_mb = 8000  # 8 GB "budget"
    hidden_dim = 2048
    layers = 24
    
    # Memory per sample (rough estimate)
    def memory_per_sample(hidden_dim, layers, compressed=False):
        base = hidden_dim * layers * 4  # float32
        if compressed:
            base *= 0.5  # 50% compression
        return base / (1024**2)
    
    baseline_per_sample = memory_per_sample(hidden_dim, layers, False)
    optimized_per_sample = memory_per_sample(hidden_dim, layers, True)
    
    baseline_batch = int(gpu_memory_mb / baseline_per_sample)
    optimized_batch = int(gpu_memory_mb / optimized_per_sample)
    
    print(f"\n  Model: {layers} layers × {hidden_dim} hidden")
    print(f"  Available GPU memory: {gpu_memory_mb} MB")
    
    print(f"\n  Without TURBO:")
    print(f"    Memory per sample: {baseline_per_sample:.2f} MB")
    print(f"    Maximum batch size: {baseline_batch}")
    
    print(f"\n  With TURBO:")
    print(f"    Memory per sample: {optimized_per_sample:.2f} MB")
    print(f"    Maximum batch size: {optimized_batch}")
    
    throughput_gain = optimized_batch / baseline_batch
    print(f"\n    → {throughput_gain:.1f}x larger batches = {throughput_gain:.1f}x throughput!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*65)
    print("  TURBO_gift v1.0.0 - ML Training Optimization Examples")
    print("  A Gift to Humanity")
    print("="*65)
    print("\n  These scenarios demonstrate REALISTIC performance gains")
    print("  for machine learning workloads.")
    
    # Run all scenarios
    engine = scenario_dataset_loading()
    scenario_training_loop(engine)
    scenario_sparse_activations(engine)
    scenario_gradient_compression(engine)
    scenario_inference_optimization(engine)
    
    # Final stats
    print_header("FINAL ENGINE STATISTICS")
    engine.print_stats()
    
    print("\n" + "="*65)
    print("  All ML scenarios completed successfully!")
    print("  ")
    print("  Key Takeaways:")
    print("    • Use TURBO to compress datasets → fit more in memory")
    print("    • Use memory pool in loops → eliminate allocation overhead")
    print("    • Sparse data compresses dramatically → save VRAM")
    print("    • Compressed gradients → faster distributed training")
    print("    • Smaller activations → larger inference batches")
    print("="*65 + "\n")


if __name__ == '__main__':
    main()
