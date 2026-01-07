"""
TURBO_gift
==========

A free, open-source GPU/CPU optimization toolkit designed to maximize
hardware utilization without requiring new equipment.

This is a gift to the world from ML Innovations.
Use it freely. Share it widely. Optimize everything.

Quick Start:
    from turbo_gift import TurboEngine, turbo_optimize
    
    # Simple usage
    optimized, metadata = turbo_optimize(your_data)
    
    # Full control
    engine = TurboEngine()
    optimized, metadata = engine.optimize(your_data)
    engine.print_stats()

License: MIT
"""

from core.engine import (
    # Core classes
    TurboEngine,
    TurboConfig,
    MemoryPool,
    PatternFilter,
    DataCompactor,
    SmartScheduler,
    AdaptiveLearner,
    MathKernel,
    
    # Enums
    OptimizationLevel,
    PrecisionMode,
    PatternType,
    QualityClass,
    
    # Convenience functions
    get_engine,
    turbo_optimize,
    turbo_stats,
    get_buffer,
    release_buffer,
    
    # Version info
    __version__,
    __author__,
    __license__,
)

# Try to import optional integrations
try:
    from core.engine import TurboTensor
except ImportError:
    TurboTensor = None

try:
    from core.engine import TurboTFTensor
except ImportError:
    TurboTFTensor = None


__all__ = [
    # Core
    'TurboEngine',
    'TurboConfig',
    'MemoryPool',
    'PatternFilter',
    'DataCompactor',
    'SmartScheduler',
    'AdaptiveLearner',
    'MathKernel',
    
    # Enums
    'OptimizationLevel',
    'PrecisionMode',
    'PatternType',
    'QualityClass',
    
    # Functions
    'get_engine',
    'turbo_optimize',
    'turbo_stats',
    'get_buffer',
    'release_buffer',
    
    # Meta
    '__version__',
    '__author__',
    '__license__',
]
