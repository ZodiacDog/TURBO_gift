"""
TURBO_gift Afterburner Interface
================================

An intuitive control panel for tuning hardware optimization settings,
inspired by MSI Afterburner's approach to making complex settings accessible.

This provides both a terminal-based UI and a web-based dashboard
for real-time monitoring and adjustment of TURBO_gift parameters.

License: MIT (Free for any use)
Author: M.L. McKnight / ML Innovations - A Gift to Humanity
"""

import sys
import os
import time
import threading
import json
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import (
    TurboEngine, TurboConfig, OptimizationLevel, PrecisionMode,
    QualityClass, turbo_stats, __version__
)

import numpy as np


# =============================================================================
# PROFILE SYSTEM
# =============================================================================

@dataclass
class TurboProfile:
    """
    Saved configuration profile for quick switching.
    """
    name: str
    description: str
    optimization_level: str = "BALANCED"
    precision_mode: str = "ADAPTIVE"
    enable_pattern_filter: bool = True
    enable_compaction: bool = True
    enable_memory_pool: bool = True
    enable_adaptive_learning: bool = True
    target_utilization: float = 0.85
    memory_budget_mb: float = 512.0
    quality_threshold: str = "LOW"
    
    def to_config(self) -> TurboConfig:
        """Convert profile to TurboConfig."""
        return TurboConfig(
            optimization_level=OptimizationLevel[self.optimization_level],
            precision_mode=PrecisionMode[self.precision_mode],
            enable_pattern_filter=self.enable_pattern_filter,
            enable_compaction=self.enable_compaction,
            enable_memory_pool=self.enable_memory_pool,
            enable_adaptive_learning=self.enable_adaptive_learning,
            target_utilization=self.target_utilization,
            memory_budget_mb=self.memory_budget_mb,
            quality_threshold=QualityClass[self.quality_threshold]
        )
    
    @classmethod
    def from_config(cls, config: TurboConfig, name: str, description: str) -> 'TurboProfile':
        """Create profile from TurboConfig."""
        return cls(
            name=name,
            description=description,
            optimization_level=config.optimization_level.name,
            precision_mode=config.precision_mode.name,
            enable_pattern_filter=config.enable_pattern_filter,
            enable_compaction=config.enable_compaction,
            enable_memory_pool=config.enable_memory_pool,
            enable_adaptive_learning=config.enable_adaptive_learning,
            target_utilization=config.target_utilization,
            memory_budget_mb=config.memory_budget_mb or 512.0,
            quality_threshold=config.quality_threshold.name
        )


# Default profiles
DEFAULT_PROFILES = {
    'safe': TurboProfile(
        name='Safe Mode',
        description='Conservative settings for stability',
        optimization_level='CONSERVATIVE',
        precision_mode='FP32',
        target_utilization=0.70,
        quality_threshold='MEDIUM'
    ),
    'balanced': TurboProfile(
        name='Balanced',
        description='Good balance of performance and stability',
        optimization_level='BALANCED',
        precision_mode='ADAPTIVE',
        target_utilization=0.85,
        quality_threshold='LOW'
    ),
    'performance': TurboProfile(
        name='Performance',
        description='Maximum performance, may reduce precision',
        optimization_level='AGGRESSIVE',
        precision_mode='FP16',
        target_utilization=0.95,
        quality_threshold='NOISE'
    ),
    'memory_saver': TurboProfile(
        name='Memory Saver',
        description='Minimize memory usage for large models',
        optimization_level='AGGRESSIVE',
        precision_mode='INT8',
        enable_compaction=True,
        target_utilization=0.80,
        memory_budget_mb=256.0,
        quality_threshold='LOW'
    ),
    'ml_training': TurboProfile(
        name='ML Training',
        description='Optimized for machine learning training workloads',
        optimization_level='BALANCED',
        precision_mode='FP16',
        enable_adaptive_learning=True,
        target_utilization=0.90,
        quality_threshold='LOW'
    ),
    'inference': TurboProfile(
        name='Inference',
        description='Optimized for model inference/serving',
        optimization_level='AGGRESSIVE',
        precision_mode='INT8',
        enable_pattern_filter=True,
        enable_compaction=True,
        target_utilization=0.95,
        quality_threshold='NOISE'
    )
}


# =============================================================================
# MONITORING SYSTEM
# =============================================================================

class PerformanceMonitor:
    """
    Real-time performance monitoring for TURBO_gift.
    """
    
    def __init__(self, engine: TurboEngine, sample_interval: float = 1.0):
        self.engine = engine
        self.sample_interval = sample_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._history: list = []
        self._max_history = 3600  # 1 hour at 1 sample/sec
        self._callbacks: list = []
    
    def start(self):
        """Start monitoring in background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def add_callback(self, callback: Callable[[Dict], None]):
        """Add callback to be called on each sample."""
        self._callbacks.append(callback)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                stats = self.engine.get_stats()
                sample = {
                    'timestamp': time.time(),
                    'stats': stats
                }
                
                self._history.append(sample)
                if len(self._history) > self._max_history:
                    self._history.pop(0)
                
                for callback in self._callbacks:
                    try:
                        callback(sample)
                    except Exception as e:
                        print(f"Monitor callback error: {e}")
                
            except Exception as e:
                print(f"Monitor error: {e}")
            
            time.sleep(self.sample_interval)
    
    def get_history(self, last_n: Optional[int] = None) -> list:
        """Get monitoring history."""
        if last_n:
            return self._history[-last_n:]
        return self._history.copy()
    
    def get_current(self) -> Optional[Dict]:
        """Get most recent sample."""
        return self._history[-1] if self._history else None
    
    def get_averages(self, window: int = 60) -> Dict:
        """Get averages over last N samples."""
        recent = self._history[-window:] if self._history else []
        
        if not recent:
            return {}
        
        # Extract key metrics
        metrics = {
            'optimizations_per_second': [],
            'memory_pool_hit_rate': [],
            'filter_rate': [],
            'compression_ratio': []
        }
        
        for sample in recent:
            stats = sample['stats']
            metrics['optimizations_per_second'].append(
                stats['engine']['optimizations_per_second']
            )
            metrics['memory_pool_hit_rate'].append(
                stats['memory_pool']['hit_rate']
            )
            metrics['filter_rate'].append(
                stats['pattern_filter']['filter_rate']
            )
            metrics['compression_ratio'].append(
                stats['data_compactor']['compression_ratio']
            )
        
        return {
            key: np.mean(values) if values else 0.0
            for key, values in metrics.items()
        }


# =============================================================================
# TERMINAL UI
# =============================================================================

class AfterburnerTerminal:
    """
    Terminal-based control panel for TURBO_gift.
    
    Provides:
        - Real-time statistics display
        - Profile switching
        - Parameter adjustment
        - Performance graphs (ASCII)
    """
    
    LOGO = """
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║   ████████╗██╗   ██╗██████╗ ██████╗  ██████╗                  ║
║   ╚══██╔══╝██║   ██║██╔══██╗██╔══██╗██╔═══██╗                 ║
║      ██║   ██║   ██║██████╔╝██████╔╝██║   ██║                 ║
║      ██║   ██║   ██║██╔══██╗██╔══██╗██║   ██║                 ║
║      ██║   ╚██████╔╝██║  ██║██████╔╝╚██████╔╝                 ║
║      ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝                  ║
║                         _gift                                  ║
║                                                                ║
║           A Gift to Humanity - Free Forever                    ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
    """
    
    def __init__(self, engine: Optional[TurboEngine] = None):
        self.engine = engine or TurboEngine()
        self.monitor = PerformanceMonitor(self.engine)
        self.current_profile = 'balanced'
        self._profiles = DEFAULT_PROFILES.copy()
    
    def clear_screen(self):
        """Clear terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print header with logo."""
        print(self.LOGO)
        print(f"  Version: {__version__}")
        print(f"  Current Profile: {self.current_profile.upper()}")
        print("")
    
    def print_stats(self):
        """Print current statistics."""
        stats = self.engine.get_stats()
        
        print("┌─────────────────────────────────────────────────────────────┐")
        print("│                    PERFORMANCE STATS                        │")
        print("├─────────────────────────────────────────────────────────────┤")
        
        # Engine stats
        eng = stats['engine']
        print(f"│  Optimizations: {eng['optimization_count']:>10,}  │  "
              f"Throughput: {eng['optimizations_per_second']:>8.1f}/s  │")
        print(f"│  Total Savings: {eng['total_savings_mb']:>10.2f} MB │  "
              f"Uptime: {eng['uptime_seconds']:>10.0f}s   │")
        
        print("├─────────────────────────────────────────────────────────────┤")
        
        # Memory pool
        pool = stats['memory_pool']
        hit_rate_bar = self._progress_bar(pool['hit_rate'], 20)
        print(f"│  Memory Pool Hit Rate: {hit_rate_bar} {pool['hit_rate']*100:>5.1f}%    │")
        print(f"│  Pool Usage: {pool['current_mb']:>8.2f} MB  │  "
              f"Buffers: {pool['pool_count']:>8}        │")
        
        print("├─────────────────────────────────────────────────────────────┤")
        
        # Compressor
        comp = stats['data_compactor']
        comp_ratio_bar = self._progress_bar(1 - comp['compression_ratio'], 20)
        print(f"│  Compression:          {comp_ratio_bar} {(1-comp['compression_ratio'])*100:>5.1f}%    │")
        print(f"│  Bytes Saved: {comp['savings_mb']:>8.2f} MB  │  "
              f"Precision Reductions: {comp['precision_reductions']:>5}│")
        
        print("├─────────────────────────────────────────────────────────────┤")
        
        # Learner
        learn = stats['learner']
        print(f"│  Adaptive Learning:                                         │")
        print(f"│    Learning Rate: {learn['learning_rate']:>6.4f}  │  "
              f"Improvement: {learn['current_improvement']:>+7.2f}%      │")
        
        print("└─────────────────────────────────────────────────────────────┘")
    
    def _progress_bar(self, value: float, width: int = 20) -> str:
        """Create ASCII progress bar."""
        filled = int(value * width)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}]"
    
    def print_profiles(self):
        """Print available profiles."""
        print("\n┌─────────────────────────────────────────────────────────────┐")
        print("│                    AVAILABLE PROFILES                        │")
        print("├─────────────────────────────────────────────────────────────┤")
        
        for key, profile in self._profiles.items():
            marker = "→" if key == self.current_profile else " "
            print(f"│ {marker} [{key:^12}] {profile.name:<20} │")
            print(f"│   {profile.description:<55} │")
        
        print("└─────────────────────────────────────────────────────────────┘")
    
    def print_sliders(self):
        """Print adjustable parameter sliders."""
        config = self.engine.config
        
        print("\n┌─────────────────────────────────────────────────────────────┐")
        print("│                    PARAMETER CONTROLS                        │")
        print("├─────────────────────────────────────────────────────────────┤")
        
        # Target utilization
        util_bar = self._progress_bar(config.target_utilization, 30)
        print(f"│  Target Utilization: {util_bar} {config.target_utilization*100:>5.0f}%│")
        
        # Memory budget
        mem_pct = (config.memory_budget_mb or 512) / 2048
        mem_bar = self._progress_bar(min(1.0, mem_pct), 30)
        print(f"│  Memory Budget:      {mem_bar} {config.memory_budget_mb or 512:>5.0f}MB│")
        
        # Feature toggles
        print("├─────────────────────────────────────────────────────────────┤")
        print(f"│  [{'X' if config.enable_pattern_filter else ' '}] Pattern Filter    "
              f"[{'X' if config.enable_compaction else ' '}] Compaction    "
              f"[{'X' if config.enable_memory_pool else ' '}] Memory Pool │")
        print(f"│  [{'X' if config.enable_adaptive_learning else ' '}] Adaptive Learning                                        │")
        
        print("├─────────────────────────────────────────────────────────────┤")
        print(f"│  Optimization: {config.optimization_level.name:<12}  │  "
              f"Precision: {config.precision_mode.name:<12}     │")
        
        print("└─────────────────────────────────────────────────────────────┘")
    
    def print_menu(self):
        """Print command menu."""
        print("\n┌─────────────────────────────────────────────────────────────┐")
        print("│  Commands:                                                   │")
        print("│    [P]rofiles  [S]ettings  [B]enchmark  [R]efresh  [Q]uit   │")
        print("│    [1-6] Quick Profile  [+/-] Adjust Utilization            │")
        print("└─────────────────────────────────────────────────────────────┘")
    
    def apply_profile(self, profile_key: str):
        """Apply a saved profile."""
        if profile_key not in self._profiles:
            print(f"Unknown profile: {profile_key}")
            return
        
        profile = self._profiles[profile_key]
        new_config = profile.to_config()
        
        # Create new engine with config
        self.engine = TurboEngine(new_config)
        self.monitor = PerformanceMonitor(self.engine)
        self.current_profile = profile_key
        
        print(f"\n✓ Applied profile: {profile.name}")
    
    def adjust_utilization(self, delta: float):
        """Adjust target utilization."""
        new_util = max(0.1, min(1.0, self.engine.config.target_utilization + delta))
        self.engine.config.target_utilization = new_util
        print(f"\n✓ Target utilization: {new_util*100:.0f}%")
    
    def run_benchmark(self):
        """Run quick benchmark to show optimization effect."""
        print("\n Running benchmark...")
        print("─" * 50)
        
        # Generate test data
        test_sizes = [
            (100, 100),
            (500, 500),
            (1000, 1000),
            (2000, 2000)
        ]
        
        results = []
        
        for size in test_sizes:
            # Create test data
            data = np.random.randn(*size).astype(np.float32)
            original_bytes = data.nbytes
            
            # Time optimization
            start = time.perf_counter()
            optimized, metadata = self.engine.optimize(data)
            opt_time = time.perf_counter() - start
            
            # Calculate metrics
            optimized_bytes = optimized.nbytes if isinstance(optimized, np.ndarray) else 0
            compression = (1 - optimized_bytes / original_bytes) * 100
            
            results.append({
                'size': size,
                'original_mb': original_bytes / (1024*1024),
                'optimized_mb': optimized_bytes / (1024*1024),
                'compression': compression,
                'time_ms': opt_time * 1000
            })
            
            print(f"  {size[0]:>4}x{size[1]:<4}: "
                  f"{compression:>5.1f}% compression, "
                  f"{opt_time*1000:>6.2f}ms")
        
        print("─" * 50)
        
        # Summary
        avg_compression = np.mean([r['compression'] for r in results])
        total_time = sum(r['time_ms'] for r in results)
        
        print(f"\n  Average compression: {avg_compression:.1f}%")
        print(f"  Total benchmark time: {total_time:.2f}ms")
        print(f"\n  Press any key to continue...")
        input()
    
    def settings_menu(self):
        """Interactive settings adjustment."""
        while True:
            self.clear_screen()
            self.print_header()
            self.print_sliders()
            
            print("\n Settings Menu:")
            print("  [1] Toggle Pattern Filter")
            print("  [2] Toggle Compaction")
            print("  [3] Toggle Memory Pool")
            print("  [4] Toggle Adaptive Learning")
            print("  [5] Cycle Optimization Level")
            print("  [6] Cycle Precision Mode")
            print("  [+] Increase Utilization")
            print("  [-] Decrease Utilization")
            print("  [B] Back to main")
            
            choice = input("\n  Enter choice: ").strip().lower()
            
            if choice == '1':
                self.engine.config.enable_pattern_filter = not self.engine.config.enable_pattern_filter
            elif choice == '2':
                self.engine.config.enable_compaction = not self.engine.config.enable_compaction
            elif choice == '3':
                self.engine.config.enable_memory_pool = not self.engine.config.enable_memory_pool
            elif choice == '4':
                self.engine.config.enable_adaptive_learning = not self.engine.config.enable_adaptive_learning
            elif choice == '5':
                levels = list(OptimizationLevel)
                current_idx = levels.index(self.engine.config.optimization_level)
                self.engine.config.optimization_level = levels[(current_idx + 1) % len(levels)]
            elif choice == '6':
                modes = list(PrecisionMode)
                current_idx = modes.index(self.engine.config.precision_mode)
                self.engine.config.precision_mode = modes[(current_idx + 1) % len(modes)]
            elif choice == '+':
                self.adjust_utilization(0.05)
            elif choice == '-':
                self.adjust_utilization(-0.05)
            elif choice == 'b':
                break
    
    def run(self):
        """Run the interactive terminal UI."""
        profile_keys = list(self._profiles.keys())
        
        while True:
            self.clear_screen()
            self.print_header()
            self.print_stats()
            self.print_menu()
            
            choice = input("\n  Enter command: ").strip().lower()
            
            if choice == 'q':
                print("\n  Goodbye! May your computations be ever optimized.")
                break
            elif choice == 'r':
                continue  # Refresh
            elif choice == 'p':
                self.print_profiles()
                profile = input("\n  Enter profile name: ").strip().lower()
                self.apply_profile(profile)
                time.sleep(1)
            elif choice == 's':
                self.settings_menu()
            elif choice == 'b':
                self.run_benchmark()
            elif choice == '+':
                self.adjust_utilization(0.05)
                time.sleep(0.5)
            elif choice == '-':
                self.adjust_utilization(-0.05)
                time.sleep(0.5)
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(profile_keys):
                    self.apply_profile(profile_keys[idx])
                    time.sleep(1)


# =============================================================================
# WEB DASHBOARD (Optional)
# =============================================================================

def create_web_dashboard(engine: TurboEngine, port: int = 8080) -> Optional[Any]:
    """
    Create a web-based dashboard for TURBO_gift.
    
    Requires: flask (optional dependency)
    
    Returns Flask app or None if Flask not available.
    """
    try:
        from flask import Flask, render_template_string, jsonify, request
    except ImportError:
        print("Flask not available. Install with: pip install flask")
        return None
    
    app = Flask(__name__)
    monitor = PerformanceMonitor(engine)
    
    DASHBOARD_HTML = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>TURBO_gift Dashboard</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Arial, sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: #eee;
                min-height: 100vh;
                padding: 20px;
            }
            .container { max-width: 1200px; margin: 0 auto; }
            .header {
                text-align: center;
                padding: 30px;
                margin-bottom: 30px;
            }
            .header h1 {
                font-size: 3em;
                background: linear-gradient(90deg, #00ff88, #00d4ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .header p { color: #888; margin-top: 10px; }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
            }
            .card {
                background: rgba(255,255,255,0.05);
                border-radius: 15px;
                padding: 25px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.1);
            }
            .card h3 {
                color: #00ff88;
                margin-bottom: 20px;
                font-size: 1.2em;
            }
            .stat {
                display: flex;
                justify-content: space-between;
                padding: 10px 0;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }
            .stat:last-child { border-bottom: none; }
            .stat-value {
                color: #00d4ff;
                font-weight: bold;
                font-size: 1.1em;
            }
            .progress-bar {
                height: 8px;
                background: rgba(255,255,255,0.1);
                border-radius: 4px;
                overflow: hidden;
                margin-top: 5px;
            }
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #00ff88, #00d4ff);
                transition: width 0.3s ease;
            }
            .profiles { margin-top: 30px; }
            .profile-btn {
                background: rgba(0,255,136,0.2);
                border: 1px solid #00ff88;
                color: #00ff88;
                padding: 10px 20px;
                border-radius: 8px;
                cursor: pointer;
                margin: 5px;
                transition: all 0.3s;
            }
            .profile-btn:hover {
                background: #00ff88;
                color: #1a1a2e;
            }
            .profile-btn.active {
                background: #00ff88;
                color: #1a1a2e;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>TURBO_gift</h1>
                <p>A Gift to Humanity - GPU/CPU Optimization Dashboard</p>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h3>⚡ Engine Performance</h3>
                    <div class="stat">
                        <span>Optimizations</span>
                        <span class="stat-value" id="opt-count">0</span>
                    </div>
                    <div class="stat">
                        <span>Throughput</span>
                        <span class="stat-value" id="throughput">0/s</span>
                    </div>
                    <div class="stat">
                        <span>Total Savings</span>
                        <span class="stat-value" id="savings">0 MB</span>
                    </div>
                </div>
                
                <div class="card">
                    <h3>🧠 Memory Pool</h3>
                    <div class="stat">
                        <span>Hit Rate</span>
                        <span class="stat-value" id="hit-rate">0%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="hit-rate-bar" style="width: 0%"></div>
                    </div>
                    <div class="stat" style="margin-top: 15px;">
                        <span>Pool Usage</span>
                        <span class="stat-value" id="pool-usage">0 MB</span>
                    </div>
                </div>
                
                <div class="card">
                    <h3>📦 Compression</h3>
                    <div class="stat">
                        <span>Compression Rate</span>
                        <span class="stat-value" id="compression">0%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="compression-bar" style="width: 0%"></div>
                    </div>
                    <div class="stat" style="margin-top: 15px;">
                        <span>Bytes Saved</span>
                        <span class="stat-value" id="bytes-saved">0 MB</span>
                    </div>
                </div>
                
                <div class="card">
                    <h3>🎯 Adaptive Learning</h3>
                    <div class="stat">
                        <span>Learning Rate</span>
                        <span class="stat-value" id="learn-rate">0.0</span>
                    </div>
                    <div class="stat">
                        <span>Current Improvement</span>
                        <span class="stat-value" id="improvement">0%</span>
                    </div>
                    <div class="stat">
                        <span>Improvements Found</span>
                        <span class="stat-value" id="improvements">0</span>
                    </div>
                </div>
            </div>
            
            <div class="card profiles">
                <h3>🔧 Quick Profiles</h3>
                <button class="profile-btn" onclick="setProfile('safe')">Safe</button>
                <button class="profile-btn active" onclick="setProfile('balanced')">Balanced</button>
                <button class="profile-btn" onclick="setProfile('performance')">Performance</button>
                <button class="profile-btn" onclick="setProfile('memory_saver')">Memory Saver</button>
                <button class="profile-btn" onclick="setProfile('ml_training')">ML Training</button>
                <button class="profile-btn" onclick="setProfile('inference')">Inference</button>
            </div>
        </div>
        
        <script>
            function updateStats() {
                fetch('/api/stats')
                    .then(r => r.json())
                    .then(data => {
                        document.getElementById('opt-count').textContent = 
                            data.engine.optimization_count.toLocaleString();
                        document.getElementById('throughput').textContent = 
                            data.engine.optimizations_per_second.toFixed(1) + '/s';
                        document.getElementById('savings').textContent = 
                            data.engine.total_savings_mb.toFixed(2) + ' MB';
                        
                        const hitRate = data.memory_pool.hit_rate * 100;
                        document.getElementById('hit-rate').textContent = hitRate.toFixed(1) + '%';
                        document.getElementById('hit-rate-bar').style.width = hitRate + '%';
                        document.getElementById('pool-usage').textContent = 
                            data.memory_pool.current_mb.toFixed(2) + ' MB';
                        
                        const compression = (1 - data.data_compactor.compression_ratio) * 100;
                        document.getElementById('compression').textContent = compression.toFixed(1) + '%';
                        document.getElementById('compression-bar').style.width = compression + '%';
                        document.getElementById('bytes-saved').textContent = 
                            data.data_compactor.savings_mb.toFixed(2) + ' MB';
                        
                        document.getElementById('learn-rate').textContent = 
                            data.learner.learning_rate.toFixed(4);
                        document.getElementById('improvement').textContent = 
                            (data.learner.current_improvement >= 0 ? '+' : '') + 
                            data.learner.current_improvement.toFixed(2) + '%';
                        document.getElementById('improvements').textContent = 
                            data.learner.improvements_found;
                    });
            }
            
            function setProfile(name) {
                fetch('/api/profile/' + name, { method: 'POST' })
                    .then(() => {
                        document.querySelectorAll('.profile-btn').forEach(b => 
                            b.classList.remove('active'));
                        event.target.classList.add('active');
                    });
            }
            
            setInterval(updateStats, 1000);
            updateStats();
        </script>
    </body>
    </html>
    """
    
    @app.route('/')
    def dashboard():
        return render_template_string(DASHBOARD_HTML)
    
    @app.route('/api/stats')
    def get_stats():
        return jsonify(engine.get_stats())
    
    @app.route('/api/profile/<name>', methods=['POST'])
    def set_profile(name):
        if name in DEFAULT_PROFILES:
            profile = DEFAULT_PROFILES[name]
            # Update engine config (simplified - full implementation would recreate engine)
            engine.config.optimization_level = OptimizationLevel[profile.optimization_level]
            engine.config.precision_mode = PrecisionMode[profile.precision_mode]
            engine.config.target_utilization = profile.target_utilization
            return jsonify({'status': 'ok', 'profile': name})
        return jsonify({'status': 'error', 'message': 'Unknown profile'}), 400
    
    return app


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main entry point for Afterburner interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='TURBO_gift Afterburner - Hardware Optimization Control Panel'
    )
    parser.add_argument('--web', action='store_true', 
                       help='Launch web dashboard instead of terminal UI')
    parser.add_argument('--port', type=int, default=8080,
                       help='Port for web dashboard')
    parser.add_argument('--profile', type=str, default='balanced',
                       choices=list(DEFAULT_PROFILES.keys()),
                       help='Initial profile to use')
    
    args = parser.parse_args()
    
    # Create engine with selected profile
    profile = DEFAULT_PROFILES[args.profile]
    engine = TurboEngine(profile.to_config())
    
    if args.web:
        app = create_web_dashboard(engine, args.port)
        if app:
            print(f"\n🚀 TURBO_gift Web Dashboard starting at http://localhost:{args.port}")
            app.run(host='0.0.0.0', port=args.port, debug=False)
        else:
            print("Web dashboard requires Flask. Install with: pip install flask")
            sys.exit(1)
    else:
        # Run terminal UI
        terminal = AfterburnerTerminal(engine)
        terminal.current_profile = args.profile
        terminal.run()


if __name__ == '__main__':
    main()
