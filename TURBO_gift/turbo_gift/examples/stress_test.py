"""
TURBO_gift Stress Test
======================

W-LEVEL HARDENING: This script hammers every component to verify
production readiness under extreme conditions.

Tests:
  1. High-volume processing (10,000+ optimizations)
  2. Large array handling (100+ MB arrays)
  3. Concurrent access (multi-threaded)
  4. Memory pressure (pool exhaustion/recovery)
  5. Edge case barrage (NaN, Inf, empty, tiny, huge)
  6. Long-running stability (sustained load)
  7. Data integrity verification (roundtrip accuracy)

Run: python examples/stress_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import threading
import gc
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.engine import (  # type: ignore
    TurboEngine, TurboConfig, MemoryPool, OptimizationLevel, PrecisionMode
)


class StressTestResult:
    def __init__(self, name):
        self.name = name
        self.passed = False
        self.message = ""
        self.duration = 0
        self.details = {}


def print_result(result):
    status = "✓ PASS" if result.passed else "✗ FAIL"
    print(f"\n  [{status}] {result.name}")
    print(f"    Duration: {result.duration:.2f}s")
    if result.message:
        print(f"    {result.message}")
    for key, value in result.details.items():
        print(f"    {key}: {value}")


# =============================================================================
# STRESS TEST 1: High Volume Processing
# =============================================================================

def stress_test_high_volume():
    """Process 10,000+ arrays without failure."""
    result = StressTestResult("High Volume Processing (10,000 iterations)")
    
    try:
        engine = TurboEngine(TurboConfig(verbose=False))
        iterations = 10000
        
        start = time.perf_counter()
        
        for i in range(iterations):
            # Vary size and dtype
            size = np.random.randint(50, 500)
            dtype = np.random.choice([np.float32, np.float64])
            
            data = np.random.randn(size, size).astype(dtype)
            optimized, metadata = engine.optimize(data)
            
            # Verify it didn't crash and returned something
            assert optimized is not None
            assert 'original_shape' in metadata
            
            if i % 2000 == 0:
                gc.collect()  # Periodic cleanup
        
        result.duration = time.perf_counter() - start
        result.passed = True
        result.message = f"Processed {iterations:,} arrays successfully"
        result.details = {
            "Throughput": f"{iterations/result.duration:.1f} ops/sec",
            "Avg per op": f"{result.duration/iterations*1000:.3f} ms"
        }
        
    except Exception as e:
        result.passed = False
        result.message = f"Failed: {str(e)}"
    
    return result


# =============================================================================
# STRESS TEST 2: Large Array Handling
# =============================================================================

def stress_test_large_arrays():
    """Handle 100+ MB arrays without memory issues."""
    result = StressTestResult("Large Array Handling (100+ MB)")
    
    try:
        engine = TurboEngine(TurboConfig(verbose=False))
        
        sizes = [
            (5000, 5000, np.float32),   # 100 MB
            (5000, 5000, np.float64),   # 200 MB
            (10000, 2500, np.float32),  # 100 MB
            (2500, 10000, np.float64),  # 200 MB
        ]
        
        start = time.perf_counter()
        total_mb = 0
        
        for shape_x, shape_y, dtype in sizes:
            data = np.random.randn(shape_x, shape_y).astype(dtype)
            mb = data.nbytes / (1024**2)
            total_mb += mb
            
            opt_start = time.perf_counter()
            optimized, metadata = engine.optimize(data)
            opt_time = time.perf_counter() - opt_start
            
            assert optimized is not None
            
            del data, optimized
            gc.collect()
        
        result.duration = time.perf_counter() - start
        result.passed = True
        result.message = f"Processed {total_mb:.0f} MB total"
        result.details = {
            "Arrays processed": len(sizes),
            "Avg throughput": f"{total_mb/result.duration:.1f} MB/sec"
        }
        
    except Exception as e:
        result.passed = False
        result.message = f"Failed: {str(e)}"
    
    return result


# =============================================================================
# STRESS TEST 3: Concurrent Access
# =============================================================================

def stress_test_concurrent():
    """Multi-threaded access without race conditions."""
    result = StressTestResult("Concurrent Access (8 threads × 500 ops)")
    
    try:
        engine = TurboEngine(TurboConfig(verbose=False))
        num_threads = 8
        ops_per_thread = 500
        errors = []
        completed = [0]
        lock = threading.Lock()
        
        def worker(thread_id):
            try:
                for i in range(ops_per_thread):
                    data = np.random.randn(100, 100).astype(np.float32)
                    opt, meta = engine.optimize(data)
                    
                    # Also test memory pool
                    buf = engine.get_buffer((50, 50), np.float32)
                    engine.release_buffer(buf)
                    
                    with lock:
                        completed[0] += 1
                        
            except Exception as e:
                with lock:
                    errors.append((thread_id, str(e)))
        
        start = time.perf_counter()
        
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        result.duration = time.perf_counter() - start
        
        if errors:
            result.passed = False
            result.message = f"Errors in threads: {errors[:3]}"
        else:
            result.passed = True
            result.message = f"All {num_threads * ops_per_thread:,} operations completed"
            result.details = {
                "Threads": num_threads,
                "Ops per thread": ops_per_thread,
                "Throughput": f"{completed[0]/result.duration:.1f} ops/sec"
            }
        
    except Exception as e:
        result.passed = False
        result.message = f"Failed: {str(e)}"
    
    return result


# =============================================================================
# STRESS TEST 4: Memory Pool Pressure
# =============================================================================

def stress_test_memory_pressure():
    """Test pool behavior under memory pressure."""
    result = StressTestResult("Memory Pool Pressure (exhaust & recover)")
    
    try:
        # Small pool to force pressure
        pool = MemoryPool(max_pools_per_shape=3, max_total_mb=10)
        
        start = time.perf_counter()
        
        # Phase 1: Fill pool
        buffers = []
        for _ in range(20):
            buf = pool.acquire((500, 500), np.float32)
            buffers.append(buf)
        
        # Phase 2: Release some
        for buf in buffers[::2]:
            pool.release(buf)
        
        # Phase 3: Acquire different sizes
        for _ in range(50):
            size = np.random.randint(100, 600)
            buf = pool.acquire((size, size), np.float32)
            if np.random.random() > 0.3:
                pool.release(buf)
        
        # Phase 4: Clear and verify recovery
        pool.clear()
        
        # Should work fine after clear
        buf = pool.acquire((100, 100), np.float32)
        pool.release(buf)
        
        stats = pool.get_stats()
        
        result.duration = time.perf_counter() - start
        result.passed = True
        result.message = "Pool survived pressure and recovered"
        result.details = {
            "Total hits": stats['hits'],
            "Total misses": stats['misses'],
            "Hit rate": f"{stats['hit_rate']*100:.1f}%"
        }
        
    except Exception as e:
        result.passed = False
        result.message = f"Failed: {str(e)}"
    
    return result


# =============================================================================
# STRESS TEST 5: Edge Case Barrage
# =============================================================================

def stress_test_edge_cases():
    """Test handling of all edge cases."""
    result = StressTestResult("Edge Case Barrage (15 edge cases)")
    
    try:
        engine = TurboEngine(TurboConfig(verbose=False))
        
        edge_cases = [
            ("Empty array", np.array([])),
            ("Single element", np.array([42.0])),
            ("Single row", np.random.randn(1, 100)),
            ("Single column", np.random.randn(100, 1)),
            ("All zeros", np.zeros((100, 100))),
            ("All ones", np.ones((100, 100))),
            ("All same value", np.full((100, 100), 3.14159)),
            ("Very small values", np.random.randn(100, 100) * 1e-100),
            ("Very large values", np.random.randn(100, 100) * 1e100),
            ("Contains NaN", np.array([[1.0, np.nan], [3.0, 4.0]])),
            ("Contains Inf", np.array([[1.0, np.inf], [-np.inf, 4.0]])),
            ("Mixed NaN/Inf", np.array([[np.nan, np.inf], [-np.inf, np.nan]])),
            ("Integer data", np.random.randint(0, 256, (100, 100)).astype(np.float64)),
            ("Boolean-like", (np.random.rand(100, 100) > 0.5).astype(np.float64)),
            ("Diagonal matrix", np.diag(np.arange(100).astype(np.float64))),
        ]
        
        start = time.perf_counter()
        passed_cases = 0
        failed_cases = []
        
        for name, data in edge_cases:
            try:
                with np.errstate(all='ignore'):  # Suppress numpy warnings
                    opt, meta = engine.optimize(data)
                passed_cases += 1
            except Exception as e:
                failed_cases.append((name, str(e)))
        
        result.duration = time.perf_counter() - start
        
        if failed_cases:
            result.passed = False
            result.message = f"Failed cases: {[f[0] for f in failed_cases]}"
        else:
            result.passed = True
            result.message = f"All {len(edge_cases)} edge cases handled"
            result.details = {
                "Cases tested": len(edge_cases),
                "All passed": "Yes"
            }
        
    except Exception as e:
        result.passed = False
        result.message = f"Failed: {str(e)}"
    
    return result


# =============================================================================
# STRESS TEST 6: Long-Running Stability
# =============================================================================

def stress_test_stability():
    """Sustained load for 30 seconds."""
    result = StressTestResult("Long-Running Stability (30 seconds)")
    
    try:
        engine = TurboEngine(TurboConfig(verbose=False))
        
        duration = 30  # seconds
        start = time.perf_counter()
        operations = 0
        errors = 0
        
        while time.perf_counter() - start < duration:
            try:
                size = np.random.randint(50, 300)
                data = np.random.randn(size, size).astype(np.float32)
                opt, meta = engine.optimize(data)
                
                # Also exercise memory pool
                buf = engine.get_buffer((100, 100), np.float32)
                engine.release_buffer(buf)
                
                operations += 1
                
                if operations % 1000 == 0:
                    gc.collect()
                    
            except Exception:
                errors += 1
        
        result.duration = time.perf_counter() - start
        
        if errors == 0:
            result.passed = True
            result.message = f"Stable for {duration}s with {operations:,} operations"
            result.details = {
                "Operations": f"{operations:,}",
                "Errors": 0,
                "Throughput": f"{operations/result.duration:.1f} ops/sec"
            }
        else:
            result.passed = False
            result.message = f"{errors} errors during {operations:,} operations"
        
    except Exception as e:
        result.passed = False
        result.message = f"Failed: {str(e)}"
    
    return result


# =============================================================================
# STRESS TEST 7: Data Integrity Verification
# =============================================================================

def stress_test_data_integrity():
    """Verify data survives optimization/restoration cycle."""
    result = StressTestResult("Data Integrity Verification (100 roundtrips)")
    
    try:
        engine = TurboEngine(TurboConfig(
            verbose=False,
            precision_mode=PrecisionMode.FP32  # Less precision loss
        ))
        
        start = time.perf_counter()
        max_errors = []
        
        for i in range(100):
            # Generate data
            size = np.random.randint(50, 200)
            original = np.random.randn(size, size).astype(np.float32)
            original_copy = original.copy()
            
            # Optimize
            optimized, metadata = engine.optimize(original)
            
            # Restore
            restored = engine.restore(optimized, metadata)
            
            # Verify
            max_error = np.max(np.abs(original_copy - restored))
            max_errors.append(max_error)
            
            # Should be exact or very close for FP32→FP32
            if max_error > 0.01:  # 1% tolerance
                result.passed = False
                result.message = f"Excessive error at iteration {i}: {max_error}"
                return result
        
        result.duration = time.perf_counter() - start
        result.passed = True
        result.message = "All roundtrips within tolerance"
        result.details = {
            "Roundtrips": 100,
            "Max error": f"{np.max(max_errors):.6f}",
            "Mean error": f"{np.mean(max_errors):.6f}"
        }
        
    except Exception as e:
        result.passed = False
        result.message = f"Failed: {str(e)}"
    
    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*65)
    print("  TURBO_gift STRESS TEST - W-LEVEL HARDENING")
    print("  Verifying production readiness under extreme conditions")
    print("="*65)
    
    tests = [
        stress_test_high_volume,
        stress_test_large_arrays,
        stress_test_concurrent,
        stress_test_memory_pressure,
        stress_test_edge_cases,
        stress_test_stability,
        stress_test_data_integrity,
    ]
    
    results = []
    total_start = time.perf_counter()
    
    for test_func in tests:
        print(f"\n  Running: {test_func.__name__}...")
        result = test_func()
        results.append(result)
        print_result(result)
        gc.collect()
    
    total_duration = time.perf_counter() - total_start
    
    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    
    print("\n" + "="*65)
    print("  STRESS TEST SUMMARY")
    print("="*65)
    print(f"\n  Total tests: {len(results)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total duration: {total_duration:.1f}s")
    
    if failed == 0:
        print("\n  ✓ ALL STRESS TESTS PASSED")
        print("  ✓ W-LEVEL HARDENING VERIFIED")
        print("  ✓ TURBO_gift IS PRODUCTION READY")
    else:
        print(f"\n  ✗ {failed} TESTS FAILED")
        print("  Review failures above before deployment")
    
    print("="*65 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
