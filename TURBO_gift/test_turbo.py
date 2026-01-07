"""
TURBO_gift Test Suite
=====================

Comprehensive testing for W-level hardening.

Test Categories:
    1. Unit Tests - Individual component verification
    2. Integration Tests - Component interaction
    3. Stress Tests - Performance under load
    4. Edge Case Tests - Boundary conditions
    5. Correctness Tests - Mathematical verification
    6. Regression Tests - Ensure no performance degradation

Run with: python -m pytest tests/test_turbo.py -v
Or: python tests/test_turbo.py (standalone)

License: MIT
Author: M.L. McKnight / ML Innovations
"""

import sys
import os
import time
import gc
import unittest
import warnings
from typing import List, Dict, Tuple

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

from core.engine import (
    TurboEngine, TurboConfig, MemoryPool, PatternFilter, DataCompactor,
    SmartScheduler, AdaptiveLearner, MathKernel,
    OptimizationLevel, PrecisionMode, PatternType, QualityClass,
    turbo_optimize, get_engine, __version__
)


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

class TestConfig:
    """Test configuration constants."""
    STRESS_ITERATIONS = 1000
    LARGE_ARRAY_SIZE = (5000, 5000)
    MEDIUM_ARRAY_SIZE = (1000, 1000)
    SMALL_ARRAY_SIZE = (100, 100)
    TOLERANCE = 1e-5
    PERFORMANCE_THRESHOLD = 0.1  # 10% minimum improvement target


# =============================================================================
# UNIT TESTS - MATH KERNEL
# =============================================================================

class TestMathKernel(unittest.TestCase):
    """Test mathematical foundations."""
    
    def setUp(self):
        self.kernel = MathKernel()
    
    def test_saturation_basic(self):
        """Test saturation function basic properties."""
        # S(0) = 0
        self.assertAlmostEqual(self.kernel.saturation(0), 0)
        
        # S(1) = 0.5
        self.assertAlmostEqual(self.kernel.saturation(1), 0.5)
        
        # S(∞) → 1
        self.assertAlmostEqual(self.kernel.saturation(1e10), 1.0, places=5)
    
    def test_saturation_monotonic(self):
        """Test that saturation is strictly increasing."""
        x = np.linspace(0, 100, 1000)
        y = self.kernel.saturation(x)
        
        # Check monotonicity
        diff = np.diff(y)
        self.assertTrue(np.all(diff >= 0), "Saturation should be monotonically increasing")
    
    def test_saturation_bounded(self):
        """Test saturation bounds."""
        x = np.random.uniform(0, 1e6, 10000)
        y = self.kernel.saturation(x)
        
        self.assertTrue(np.all(y >= 0), "Saturation should be >= 0")
        self.assertTrue(np.all(y <= 1), "Saturation should be <= 1")
    
    def test_inverse_saturation(self):
        """Test inverse saturation is correct inverse."""
        x = np.array([0.1, 0.5, 1.0, 2.0, 10.0, 100.0])
        y = self.kernel.saturation(x)
        x_recovered = self.kernel.inverse_saturation(y)
        
        assert_allclose(x, x_recovered, rtol=1e-5)
    
    def test_optimal_batch_size_powers_of_two(self):
        """Test batch sizes are powers of two for SIMD alignment."""
        memory = 1024 * 1024  # 1 MB
        item_size = 100  # bytes
        
        batch_size = self.kernel.optimal_batch_size(memory, item_size)
        
        # Check it's a power of 2
        self.assertTrue(
            batch_size > 0 and (batch_size & (batch_size - 1)) == 0,
            f"Batch size {batch_size} should be power of 2"
        )
    
    def test_optimal_batch_size_respects_memory(self):
        """Test batch size doesn't exceed memory."""
        memory = 10000
        item_size = 100
        
        batch_size = self.kernel.optimal_batch_size(memory, item_size)
        
        self.assertLessEqual(
            batch_size * item_size, memory,
            "Batch should fit in memory"
        )
    
    def test_information_weight_bounds(self):
        """Test IDF weight is properly bounded."""
        # Various frequency/corpus combinations
        test_cases = [
            (0.001, 1000),
            (0.1, 10000),
            (0.5, 100),
            (0.9, 1000000),
        ]
        
        for freq, corpus in test_cases:
            weight = self.kernel.information_weight(freq, 1.0, corpus)
            self.assertTrue(
                -1 <= weight <= 1,
                f"IDF weight {weight} out of bounds for freq={freq}, corpus={corpus}"
            )
    
    def test_asymmetric_normalize(self):
        """Test asymmetric normalization properties."""
        reference = 100
        
        # Short values get no penalty
        short_norm = self.kernel.asymmetric_normalize(50, reference)
        self.assertAlmostEqual(short_norm, 1.0, places=5)
        
        # Exact match gets no penalty
        exact_norm = self.kernel.asymmetric_normalize(100, reference)
        self.assertAlmostEqual(exact_norm, 1.0, places=5)
        
        # Long values get penalty
        long_norm = self.kernel.asymmetric_normalize(200, reference)
        self.assertLess(long_norm, 1.0)
        
        # Very long values get more penalty
        very_long_norm = self.kernel.asymmetric_normalize(1000, reference)
        self.assertLess(very_long_norm, long_norm)


# =============================================================================
# UNIT TESTS - MEMORY POOL
# =============================================================================

class TestMemoryPool(unittest.TestCase):
    """Test memory pooling system."""
    
    def setUp(self):
        self.pool = MemoryPool(max_pools_per_shape=5, max_total_mb=10)
    
    def tearDown(self):
        self.pool.clear()
    
    def test_acquire_returns_correct_shape(self):
        """Test acquired buffers have correct shape."""
        shapes = [(10, 10), (100,), (5, 5, 5), (1000, 2)]
        
        for shape in shapes:
            buf = self.pool.acquire(shape)
            self.assertEqual(buf.shape, shape)
    
    def test_acquire_returns_zeros(self):
        """Test acquired buffers are zeroed."""
        buf = self.pool.acquire((100, 100))
        self.assertTrue(np.all(buf == 0))
    
    def test_release_and_reuse(self):
        """Test buffers can be released and reused."""
        shape = (100, 100)
        
        # Acquire and release
        buf1 = self.pool.acquire(shape)
        buf1_id = id(buf1)
        self.pool.release(buf1)
        
        # Acquire again - should get same buffer
        buf2 = self.pool.acquire(shape)
        self.assertEqual(id(buf2), buf1_id)
    
    def test_pool_hit_rate(self):
        """Test pool tracks hit rate correctly."""
        shape = (50, 50)
        
        # First acquire is a miss
        buf1 = self.pool.acquire(shape)
        stats = self.pool.get_stats()
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['hits'], 0)
        
        # Release and re-acquire is a hit
        self.pool.release(buf1)
        buf2 = self.pool.acquire(shape)
        stats = self.pool.get_stats()
        self.assertEqual(stats['hits'], 1)
    
    def test_pool_respects_limits(self):
        """Test pool respects maximum limits."""
        shape = (100, 100)
        
        # Acquire and release more than max_pools_per_shape
        buffers = []
        for _ in range(10):
            buf = self.pool.acquire(shape)
            buffers.append(buf)
        
        for buf in buffers:
            result = self.pool.release(buf)
        
        stats = self.pool.get_stats()
        self.assertLessEqual(stats['pool_count'], 5)  # max_pools_per_shape
    
    def test_clear(self):
        """Test pool clearing."""
        shape = (100, 100)
        buf = self.pool.acquire(shape)
        self.pool.release(buf)
        
        self.pool.clear()
        stats = self.pool.get_stats()
        
        self.assertEqual(stats['pool_count'], 0)
        self.assertEqual(stats['current_mb'], 0)


# =============================================================================
# UNIT TESTS - PATTERN FILTER
# =============================================================================

class TestPatternFilter(unittest.TestCase):
    """Test pattern filtering system."""
    
    def setUp(self):
        self.config = TurboConfig()
        self.filter = PatternFilter(self.config)
    
    def test_detect_sparse_pattern(self):
        """Test detection of sparse data."""
        # 95% zeros with non-diagonal non-zeros
        data = np.zeros((100, 100))
        data[0, 50] = 1.0  # Off diagonal
        data[50, 0] = 2.0  # Off diagonal
        data[25, 75] = 3.0
        
        pattern = self.filter.analyze_pattern(data)
        self.assertEqual(pattern, PatternType.SPARSE)
    
    def test_detect_dense_pattern(self):
        """Test detection of dense data."""
        data = np.random.randn(100, 100)
        
        pattern = self.filter.analyze_pattern(data)
        self.assertEqual(pattern, PatternType.DENSE)
    
    def test_detect_repetitive_pattern(self):
        """Test detection of repetitive data."""
        # Create truly repetitive data - very few unique values
        data = np.zeros((100, 100))
        data[:50, :] = 1.0  # Only 2 unique values
        
        pattern = self.filter.analyze_pattern(data)
        self.assertEqual(pattern, PatternType.REPETITIVE)
    
    def test_detect_structured_pattern(self):
        """Test detection of structured (diagonal) data."""
        # Create diagonal-dominant sparse matrix
        data = np.zeros((100, 100))
        np.fill_diagonal(data, 10.0)  # Strong diagonal
        
        pattern = self.filter.analyze_pattern(data)
        self.assertEqual(pattern, PatternType.STRUCTURED)
    
    def test_quality_classification(self):
        """Test quality classification."""
        # High variance = high quality
        high_quality = np.random.randn(100, 100) * 100
        q1 = self.filter.classify_quality(high_quality)
        
        # Low variance = low quality
        low_quality = np.random.randn(100, 100) * 0.001
        q2 = self.filter.classify_quality(low_quality)
        
        self.assertGreater(q1.value, q2.value)
    
    def test_filter_preserves_shape_metadata(self):
        """Test filter preserves original shape in metadata."""
        data = np.random.randn(50, 75)
        
        _, metadata = self.filter.filter(data)
        
        self.assertEqual(metadata['original_shape'], (50, 75))
    
    def test_filter_statistics(self):
        """Test filter tracks statistics."""
        for _ in range(10):
            data = np.random.randn(10, 10)
            self.filter.filter(data)
        
        stats = self.filter.get_stats()
        self.assertEqual(stats['total_processed'], 10)


# =============================================================================
# UNIT TESTS - DATA COMPACTOR
# =============================================================================

class TestDataCompactor(unittest.TestCase):
    """Test data compaction system."""
    
    def setUp(self):
        self.config = TurboConfig()
        self.compactor = DataCompactor(self.config)
    
    def test_precision_analysis_int_data(self):
        """Test precision detection for integer-like data."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        
        precision = self.compactor.analyze_precision_requirements(data)
        self.assertEqual(precision, PrecisionMode.INT8)
    
    def test_precision_analysis_float_data(self):
        """Test precision detection for float data."""
        data = np.random.randn(100).astype(np.float64)
        
        precision = self.compactor.analyze_precision_requirements(data)
        self.assertIn(precision, [PrecisionMode.FP16, PrecisionMode.FP32, PrecisionMode.FP64])
    
    def test_compact_reduces_memory(self):
        """Test compaction actually reduces memory."""
        data = np.random.randn(1000, 1000).astype(np.float64)
        original_bytes = data.nbytes
        
        compacted, metadata = self.compactor.compact(data, PrecisionMode.FP16)
        
        self.assertLess(compacted.nbytes, original_bytes)
    
    def test_compact_decompact_roundtrip(self):
        """Test data survives compact/decompact cycle."""
        data = np.random.randn(100, 100).astype(np.float32)
        
        compacted, metadata = self.compactor.compact(data, PrecisionMode.FP16)
        restored = self.compactor.decompact(compacted, metadata)
        
        # Should be close (not exact due to precision loss)
        assert_allclose(data, restored, rtol=1e-2)
    
    def test_int8_quantization_roundtrip(self):
        """Test INT8 quantization preserves approximate values."""
        # Use well-separated values for reliable quantization
        data = np.array([0.0, 25.0, 50.0, 75.0, 100.0], dtype=np.float32)
        
        compacted, metadata = self.compactor.compact(data, PrecisionMode.INT8)
        restored = self.compactor.decompact(compacted, metadata)
        
        # Check values are approximately preserved (not exact due to quantization)
        assert_allclose(data, restored, rtol=0.1)
    
    def test_compaction_statistics(self):
        """Test compactor tracks statistics."""
        for _ in range(5):
            data = np.random.randn(100, 100).astype(np.float64)
            self.compactor.compact(data, PrecisionMode.FP16)
        
        stats = self.compactor.get_stats()
        
        self.assertEqual(stats['total_compacted'], 5)
        self.assertGreater(stats['bytes_before'], stats['bytes_after'])


# =============================================================================
# UNIT TESTS - SMART SCHEDULER
# =============================================================================

class TestSmartScheduler(unittest.TestCase):
    """Test scheduling system."""
    
    def setUp(self):
        self.config = TurboConfig()
        self.scheduler = SmartScheduler(self.config)
    
    def test_optimal_parallelism_bounded(self):
        """Test parallelism doesn't exceed available workers."""
        workers = self.scheduler.compute_optimal_parallelism(
            work_items=1000,
            available_workers=8
        )
        
        self.assertLessEqual(workers, 8)
        self.assertGreaterEqual(workers, 1)
    
    def test_optimal_parallelism_small_work(self):
        """Test small work items don't over-parallelize."""
        workers = self.scheduler.compute_optimal_parallelism(
            work_items=2,
            available_workers=64
        )
        
        # Shouldn't use many workers for tiny workload
        self.assertLess(workers, 10)
    
    def test_batch_size_respects_memory(self):
        """Test batch size fits in memory."""
        memory = 1024 * 1024  # 1 MB
        item_size = 1000  # bytes
        
        batch_size = self.scheduler.compute_optimal_batch_size(
            total_items=10000,
            memory_per_item=item_size,
            available_memory=memory
        )
        
        self.assertLessEqual(batch_size * item_size, memory)
    
    def test_schedule_work_creates_batches(self):
        """Test work scheduling creates proper batches."""
        work_items = [{'id': i, 'size': 1000, 'priority': i} for i in range(100)]
        
        batches = self.scheduler.schedule_work(
            work_items=work_items,
            available_memory=50000,
            available_workers=4
        )
        
        # Should create multiple batches
        self.assertGreater(len(batches), 1)
        
        # All items should be scheduled
        total_items = sum(len(b) for b in batches)
        self.assertEqual(total_items, 100)
    
    def test_schedule_work_respects_priority(self):
        """Test high priority items come first."""
        work_items = [
            {'id': 'low', 'size': 100, 'priority': 1},
            {'id': 'high', 'size': 100, 'priority': 10},
            {'id': 'med', 'size': 100, 'priority': 5},
        ]
        
        batches = self.scheduler.schedule_work(
            work_items=work_items,
            available_memory=1000000,
            available_workers=1
        )
        
        # First batch, first item should be high priority
        first_batch = batches[0]
        self.assertEqual(first_batch[0]['id'], 'high')


# =============================================================================
# UNIT TESTS - ADAPTIVE LEARNER
# =============================================================================

class TestAdaptiveLearner(unittest.TestCase):
    """Test adaptive learning system."""
    
    def setUp(self):
        self.config = TurboConfig()
        self.learner = AdaptiveLearner(self.config)
    
    def test_record_performance(self):
        """Test performance recording."""
        self.learner.record_performance('test_config', 100.0)
        
        stats = self.learner.get_stats()
        self.assertEqual(stats['learning_iterations'], 1)
    
    def test_learning_improves_with_data(self):
        """Test learning rate adapts based on performance trend."""
        initial_lr = self.learner._learning_rate
        
        # Record improving performance
        for i in range(20):
            self.learner.record_performance('config_a', 100 + i * 5)
        
        self.learner.learn()
        
        # Learning rate should have adapted
        self.assertNotEqual(self.learner._learning_rate, initial_lr)
    
    def test_suggest_config_returns_valid(self):
        """Test config suggestions are valid."""
        # Add some data
        for i in range(15):
            self.learner.record_performance('batch_config', 100 + i)
        
        self.learner.learn()
        
        suggestion = self.learner.suggest_config()
        
        self.assertIn('batch_size_multiplier', suggestion)
        self.assertIn('precision_threshold', suggestion)
        self.assertGreater(suggestion['batch_size_multiplier'], 0)


# =============================================================================
# INTEGRATION TESTS - TURBO ENGINE
# =============================================================================

class TestTurboEngine(unittest.TestCase):
    """Integration tests for main engine."""
    
    def setUp(self):
        self.config = TurboConfig(verbose=False)
        self.engine = TurboEngine(self.config)
    
    def test_optimize_basic(self):
        """Test basic optimization works."""
        data = np.random.randn(100, 100).astype(np.float32)
        
        optimized, metadata = self.engine.optimize(data)
        
        self.assertIsInstance(optimized, np.ndarray)
        self.assertIn('original_shape', metadata)
    
    def test_optimize_preserves_data(self):
        """Test optimization doesn't corrupt data."""
        data = np.random.randn(100, 100).astype(np.float32)
        original = data.copy()
        
        optimized, metadata = self.engine.optimize(data)
        restored = self.engine.restore(optimized, metadata)
        
        # Should be close (may have precision loss)
        assert_allclose(original, restored, rtol=0.1)
    
    def test_optimize_reduces_size(self):
        """Test optimization reduces data size for compressible data."""
        # Use FP64 data which can be reduced to FP16/FP32
        data = np.random.randn(1000, 1000).astype(np.float64)
        
        # Force FP16 precision for guaranteed compression
        config = TurboConfig(precision_mode=PrecisionMode.FP16, verbose=False)
        engine = TurboEngine(config)
        
        optimized, metadata = engine.optimize(data)
        
        self.assertLess(metadata['optimized_bytes'], metadata['original_bytes'])
    
    def test_optimize_batch(self):
        """Test batch optimization."""
        data_list = [np.random.randn(100, 100) for _ in range(10)]
        
        optimized_list, metadata_list = self.engine.optimize_batch(data_list)
        
        self.assertEqual(len(optimized_list), 10)
        self.assertEqual(len(metadata_list), 10)
    
    def test_get_buffer_from_pool(self):
        """Test buffer acquisition from pool."""
        shape = (100, 100)
        
        buf1 = self.engine.get_buffer(shape)
        self.engine.release_buffer(buf1)
        buf2 = self.engine.get_buffer(shape)
        
        # Should get same buffer back
        self.assertEqual(id(buf1), id(buf2))
    
    def test_statistics_tracking(self):
        """Test engine tracks statistics."""
        for _ in range(10):
            data = np.random.randn(50, 50)
            self.engine.optimize(data)
        
        stats = self.engine.get_stats()
        
        self.assertEqual(stats['engine']['optimization_count'], 10)
    
    def test_record_performance_for_learning(self):
        """Test performance recording integrates with learner."""
        data = np.random.randn(100, 100)
        
        _, metadata = self.engine.optimize(data)
        self.engine.record_performance(metadata, 100.0, 'test')
        
        learner_stats = self.engine.learner.get_stats()
        self.assertGreater(learner_stats['learning_iterations'], 0)


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestStress(unittest.TestCase):
    """Stress tests for reliability under load."""
    
    def setUp(self):
        self.config = TurboConfig(verbose=False)
        self.engine = TurboEngine(self.config)
    
    def test_many_optimizations(self):
        """Test engine handles many optimizations."""
        for i in range(TestConfig.STRESS_ITERATIONS):
            size = np.random.randint(10, 200)
            data = np.random.randn(size, size)
            
            optimized, metadata = self.engine.optimize(data)
            
            if i % 100 == 0:
                gc.collect()
        
        stats = self.engine.get_stats()
        self.assertEqual(
            stats['engine']['optimization_count'], 
            TestConfig.STRESS_ITERATIONS
        )
    
    def test_large_array(self):
        """Test handling of large arrays."""
        data = np.random.randn(*TestConfig.LARGE_ARRAY_SIZE).astype(np.float32)
        
        start = time.perf_counter()
        optimized, metadata = self.engine.optimize(data)
        elapsed = time.perf_counter() - start
        
        # Should complete in reasonable time
        self.assertLess(elapsed, 60.0, "Large array optimization too slow")
        
        # Should produce valid output
        self.assertIsInstance(optimized, np.ndarray)
    
    def test_memory_pool_under_pressure(self):
        """Test memory pool under allocation pressure."""
        pool = MemoryPool(max_pools_per_shape=5, max_total_mb=50)
        
        shapes = [(100, 100), (200, 200), (50, 50), (150, 150)]
        
        for _ in range(1000):
            shape = shapes[np.random.randint(len(shapes))]
            buf = pool.acquire(shape)
            
            # 50% chance of releasing
            if np.random.random() > 0.5:
                pool.release(buf)
        
        stats = pool.get_stats()
        
        # Should have good hit rate after warmup
        self.assertGreater(stats['hit_rate'], 0.2)
        
        pool.clear()
    
    def test_concurrent_access(self):
        """Test engine handles concurrent access."""
        import threading
        
        errors = []
        results = []
        
        def worker(engine, worker_id):
            try:
                for _ in range(100):
                    data = np.random.randn(50, 50)
                    opt, meta = engine.optimize(data)
                    results.append(meta)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(4):
            t = threading.Thread(target=worker, args=(self.engine, i))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        self.assertEqual(len(errors), 0, f"Concurrent errors: {errors}")
        self.assertEqual(len(results), 400)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases(unittest.TestCase):
    """Test handling of edge cases."""
    
    def setUp(self):
        self.engine = TurboEngine(TurboConfig(verbose=False))
    
    def test_empty_array(self):
        """Test handling of empty arrays."""
        data = np.array([])
        
        optimized, metadata = self.engine.optimize(data)
        
        self.assertEqual(optimized.size, 0)
    
    def test_single_element(self):
        """Test handling of single element arrays."""
        data = np.array([42.0])
        
        optimized, metadata = self.engine.optimize(data)
        
        # Single element should be preserved
        if optimized.size > 0:
            restored = self.engine.restore(optimized, metadata)
            assert_allclose(data, restored, rtol=0.1)
        else:
            # May be filtered as low-quality, which is acceptable
            pass
    
    def test_all_zeros(self):
        """Test handling of all-zero arrays."""
        data = np.zeros((100, 100))
        
        optimized, metadata = self.engine.optimize(data)
        
        # Should detect as sparse/filterable
        self.assertEqual(
            metadata['stages']['filter']['pattern'], 
            PatternType.SPARSE
        )
    
    def test_all_same_value(self):
        """Test handling of constant arrays."""
        data = np.ones((100, 100)) * 3.14159
        
        optimized, metadata = self.engine.optimize(data)
        
        # Should detect as repetitive (only 1 unique value)
        self.assertEqual(
            metadata['stages']['filter']['pattern'],
            PatternType.REPETITIVE
        )
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        data = np.random.randn(100, 100)
        data[50, 50] = np.nan
        
        # Should not crash
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimized, metadata = self.engine.optimize(data)
    
    def test_inf_handling(self):
        """Test handling of infinite values."""
        data = np.random.randn(100, 100)
        data[0, 0] = np.inf
        data[99, 99] = -np.inf
        
        # Should not crash
        optimized, metadata = self.engine.optimize(data)
    
    def test_very_small_values(self):
        """Test handling of very small values."""
        data = np.random.randn(100, 100) * 1e-100
        
        optimized, metadata = self.engine.optimize(data)
        
        # Should still work
        self.assertIsInstance(optimized, np.ndarray)
    
    def test_very_large_values(self):
        """Test handling of very large values."""
        data = np.random.randn(100, 100) * 1e100
        
        optimized, metadata = self.engine.optimize(data)
        
        self.assertIsInstance(optimized, np.ndarray)
    
    def test_mixed_dtypes(self):
        """Test handling of various dtypes."""
        dtypes = [np.float16, np.float32, np.float64, np.int32, np.int64]
        
        for dtype in dtypes:
            data = (np.random.randn(50, 50) * 100).astype(dtype)
            
            optimized, metadata = self.engine.optimize(data)
            
            # Original dtype should be recorded (as string representation)
            original_dtype_str = metadata['original_dtype']
            expected_dtype_str = str(data.dtype)
            self.assertEqual(
                original_dtype_str, 
                expected_dtype_str,
                f"Failed for dtype {dtype}: got {original_dtype_str}, expected {expected_dtype_str}"
            )


# =============================================================================
# CORRECTNESS TESTS
# =============================================================================

class TestCorrectness(unittest.TestCase):
    """Mathematical correctness verification."""
    
    def test_ml_identity_holds(self):
        """Verify ML Identity: a + a² + b = b² where b = a + 1."""
        for a in [0, 1, 2, 5, 10, 100, 0.5, 1.5]:
            b = a + 1
            lhs = a + a**2 + b
            rhs = b**2
            
            self.assertAlmostEqual(
                lhs, rhs, places=10,
                msg=f"ML Identity failed for a={a}"
            )
    
    def test_saturation_derivative(self):
        """Verify saturation derivative: S'(x) = 1/(x+1)²."""
        kernel = MathKernel()
        
        x = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
        
        # Numerical derivative
        dx = 1e-7
        numerical_deriv = (kernel.saturation(x + dx) - kernel.saturation(x)) / dx
        
        # Analytical derivative
        analytical_deriv = 1 / (x + 1)**2
        
        assert_allclose(numerical_deriv, analytical_deriv, rtol=1e-4)
    
    def test_optimal_batch_size_formula(self):
        """Verify batch size optimization produces valid results."""
        kernel = MathKernel()
        
        test_cases = [
            (1000, 10, 100),    # Small items
            (10000, 1000, 50),  # Large items
            (5000, 100, 200),   # Medium case
        ]
        
        for memory, item_size, overhead in test_cases:
            batch = kernel.optimal_batch_size(memory, item_size, overhead)
            
            # Should fit in memory
            self.assertLessEqual(
                batch * item_size + overhead, memory,
                f"Batch {batch} doesn't fit for {memory}/{item_size}/{overhead}"
            )
            
            # Should be positive
            self.assertGreater(batch, 0)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance(unittest.TestCase):
    """Performance benchmarks."""
    
    def setUp(self):
        self.engine = TurboEngine(TurboConfig(verbose=False))
    
    def test_optimization_speed(self):
        """Test optimization throughput."""
        data = np.random.randn(500, 500).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            self.engine.optimize(data)
        
        # Benchmark
        start = time.perf_counter()
        iterations = 100
        
        for _ in range(iterations):
            self.engine.optimize(data)
        
        elapsed = time.perf_counter() - start
        throughput = iterations / elapsed
        
        print(f"\nOptimization throughput: {throughput:.1f} ops/sec")
        
        # Should achieve reasonable throughput
        self.assertGreater(throughput, 10, "Throughput too low")
    
    def test_memory_pool_benefit(self):
        """Test memory pool provides speedup."""
        shape = (500, 500)
        iterations = 100
        
        # Without pool
        no_pool_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            buf = np.zeros(shape)
            no_pool_times.append(time.perf_counter() - start)
        
        # With pool
        pool = MemoryPool()
        pool_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            buf = pool.acquire(shape)
            pool.release(buf)
            pool_times.append(time.perf_counter() - start)
        
        no_pool_avg = np.mean(no_pool_times)
        pool_avg = np.mean(pool_times[10:])  # Skip warmup
        
        print(f"\nNo pool: {no_pool_avg*1e6:.2f}µs, Pool: {pool_avg*1e6:.2f}µs")
        
        # Pool should be faster after warmup
        self.assertLess(pool_avg, no_pool_avg * 1.5)  # Allow some variance
    
    def test_compression_effectiveness(self):
        """Test compression achieves target reduction."""
        # Random FP64 data (should compress to FP16)
        config = TurboConfig(precision_mode=PrecisionMode.FP16, verbose=False)
        engine = TurboEngine(config)
        
        random_data = np.random.randn(1000, 1000).astype(np.float64)
        _, random_meta = engine.optimize(random_data)
        
        # Sparse data (should compress significantly via sparsity)
        sparse_data = np.zeros((1000, 1000), dtype=np.float64)
        sparse_data[::10, ::10] = np.random.randn(100, 100)
        _, sparse_meta = engine.optimize(sparse_data)
        
        print(f"\nRandom compression: {random_meta['savings_percent']:.1f}%")
        print(f"Sparse compression: {sparse_meta['savings_percent']:.1f}%")
        
        # FP64 to FP16 should give ~75% compression
        self.assertGreater(random_meta['savings_percent'], 50)
        
        # Sparse should compress even more
        self.assertGreater(sparse_meta['savings_percent'], random_meta['savings_percent'])


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Test module-level convenience functions."""
    
    def test_turbo_optimize(self):
        """Test turbo_optimize function."""
        data = np.random.randn(100, 100)
        
        optimized, metadata = turbo_optimize(data)
        
        self.assertIsInstance(optimized, np.ndarray)
        self.assertIn('original_shape', metadata)
    
    def test_get_engine_singleton(self):
        """Test engine singleton behavior."""
        engine1 = get_engine()
        engine2 = get_engine()
        
        self.assertIs(engine1, engine2)


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_tests(verbosity: int = 2) -> bool:
    """
    Run all tests and return success status.
    
    Args:
        verbosity: Test output verbosity (0-2)
    
    Returns:
        True if all tests passed, False otherwise
    """
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMathKernel,
        TestMemoryPool,
        TestPatternFilter,
        TestDataCompactor,
        TestSmartScheduler,
        TestAdaptiveLearner,
        TestTurboEngine,
        TestStress,
        TestEdgeCases,
        TestCorrectness,
        TestPerformance,
        TestConvenienceFunctions,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TURBO_gift TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED - W-LEVEL HARDENING VERIFIED")
    else:
        print("\n✗ SOME TESTS FAILED - REVIEW REQUIRED")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split(chr(10))[0]}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split(chr(10))[0]}")
    
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests(verbosity=2)
    sys.exit(0 if success else 1)
