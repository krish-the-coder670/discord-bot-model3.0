"""
Performance Configuration Module
Provides hardware-specific optimization settings and utilities
"""

import os
import psutil
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class HardwareProfile:
    """Hardware profile for optimization"""
    name: str
    min_ram_gb: int
    min_cpu_cores: int
    recommended_settings: Dict[str, Any]

# Predefined hardware profiles
HARDWARE_PROFILES = {
    'low': HardwareProfile(
        name='Low-end Hardware',
        min_ram_gb=4,
        min_cpu_cores=2,
        recommended_settings={
            'max_memory_usage': 2048,  # 2GB
            'cache_size': 200,
            'message_cache_size': 500,
            'embedding_cache_size': 1000,
            'max_workers': 1,
            'batch_size': 8,
            'max_tokens': 75,
            'temperature': 0.5,
            'enable_quantization': True,
            'enable_memory_mapping': True,
            'enable_batch_processing': False,
            'model_precision': 'int8'
        }
    ),
    'medium': HardwareProfile(
        name='Medium Hardware',
        min_ram_gb=8,
        min_cpu_cores=4,
        recommended_settings={
            'max_memory_usage': 6144,  # 6GB
            'cache_size': 1000,
            'message_cache_size': 2000,
            'embedding_cache_size': 5000,
            'max_workers': 3,
            'batch_size': 32,
            'max_tokens': 150,
            'temperature': 0.7,
            'enable_quantization': True,
            'enable_memory_mapping': True,
            'enable_batch_processing': True,
            'model_precision': 'float16'
        }
    ),
    'high': HardwareProfile(
        name='High-end Hardware',
        min_ram_gb=16,
        min_cpu_cores=8,
        recommended_settings={
            'max_memory_usage': 12288,  # 12GB
            'cache_size': 2000,
            'message_cache_size': 5000,
            'embedding_cache_size': 10000,
            'max_workers': 6,
            'batch_size': 64,
            'max_tokens': 200,
            'temperature': 0.8,
            'enable_quantization': False,
            'enable_memory_mapping': True,
            'enable_batch_processing': True,
            'model_precision': 'float32'
        }
    )
}

class PerformanceOptimizer:
    """Automatic performance optimization based on hardware detection"""
    
    def __init__(self):
        self.system_info = self._detect_hardware()
        self.recommended_profile = self._recommend_profile()
        
    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect system hardware specifications"""
        try:
            # Get RAM info
            memory = psutil.virtual_memory()
            ram_gb = memory.total / (1024**3)
            
            # Get CPU info
            cpu_count = psutil.cpu_count(logical=False)  # Physical cores
            cpu_freq = psutil.cpu_freq()
            
            # Get disk info
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / (1024**3)
            
            return {
                'ram_gb': ram_gb,
                'cpu_cores': cpu_count,
                'cpu_freq_mhz': cpu_freq.current if cpu_freq else 0,
                'disk_free_gb': disk_free_gb,
                'platform': os.name
            }
        except Exception as e:
            logger.error(f"Hardware detection error: {e}")
            return {
                'ram_gb': 8,  # Default fallback
                'cpu_cores': 4,
                'cpu_freq_mhz': 2000,
                'disk_free_gb': 50,
                'platform': 'unknown'
            }
    
    def _recommend_profile(self) -> str:
        """Recommend hardware profile based on detected specs"""
        ram_gb = self.system_info['ram_gb']
        cpu_cores = self.system_info['cpu_cores']
        
        if ram_gb >= 16 and cpu_cores >= 8:
            return 'high'
        elif ram_gb >= 8 and cpu_cores >= 4:
            return 'medium'
        else:
            return 'low'
    
    def get_optimized_config(self, override_profile: Optional[str] = None) -> Dict[str, Any]:
        """Get optimized configuration for the system"""
        profile_name = override_profile or self.recommended_profile
        profile = HARDWARE_PROFILES.get(profile_name, HARDWARE_PROFILES['medium'])
        
        config = profile.recommended_settings.copy()
        
        # Apply system-specific adjustments
        if self.system_info['ram_gb'] < 6:
            # Further reduce memory usage for very low RAM systems
            config['max_memory_usage'] = min(config['max_memory_usage'], int(self.system_info['ram_gb'] * 1024 * 0.4))
            config['cache_size'] = min(config['cache_size'], 100)
            config['message_cache_size'] = min(config['message_cache_size'], 300)
        
        if self.system_info['cpu_cores'] <= 2:
            # Reduce parallelism for low-core systems
            config['max_workers'] = 1
            config['batch_size'] = min(config['batch_size'], 16)
        
        # Add system info to config
        config['detected_hardware'] = self.system_info
        config['recommended_profile'] = profile_name
        config['profile_name'] = profile.name
        
        return config
    
    def save_config(self, filepath: str = 'performance_config.json'):
        """Save optimized configuration to file"""
        config = self.get_optimized_config()
        try:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Performance configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving performance config: {e}")
    
    def load_config(self, filepath: str = 'performance_config.json') -> Dict[str, Any]:
        """Load configuration from file or generate new one"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    config = json.load(f)
                logger.info(f"Performance configuration loaded from {filepath}")
                return config
            except Exception as e:
                logger.error(f"Error loading performance config: {e}")
        
        # Generate and save new config
        config = self.get_optimized_config()
        self.save_config(filepath)
        return config
    
    def print_system_info(self):
        """Print detected system information"""
        print("\n" + "="*50)
        print("SYSTEM HARDWARE DETECTION")
        print("="*50)
        print(f"RAM: {self.system_info['ram_gb']:.1f} GB")
        print(f"CPU Cores: {self.system_info['cpu_cores']}")
        print(f"CPU Frequency: {self.system_info['cpu_freq_mhz']:.0f} MHz")
        print(f"Free Disk Space: {self.system_info['disk_free_gb']:.1f} GB")
        print(f"Platform: {self.system_info['platform']}")
        print(f"\nRecommended Profile: {self.recommended_profile.upper()}")
        print(f"Profile Name: {HARDWARE_PROFILES[self.recommended_profile].name}")
        print("="*50)

class MemoryOptimizer:
    """Memory usage optimization utilities"""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get detailed memory usage information"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024**2),  # Resident Set Size
            'vms_mb': memory_info.vms / (1024**2),  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / (1024**2)
        }
    
    @staticmethod
    def suggest_memory_optimizations(current_usage_mb: float, max_usage_mb: float) -> List[str]:
        """Suggest memory optimization strategies"""
        suggestions = []
        usage_ratio = current_usage_mb / max_usage_mb
        
        if usage_ratio > 0.9:
            suggestions.extend([
                "Critical: Memory usage is very high",
                "- Reduce cache sizes immediately",
                "- Enable aggressive garbage collection",
                "- Consider restarting the application"
            ])
        elif usage_ratio > 0.8:
            suggestions.extend([
                "Warning: Memory usage is high",
                "- Reduce message cache size",
                "- Clear old conversation history",
                "- Enable memory cleanup"
            ])
        elif usage_ratio > 0.6:
            suggestions.extend([
                "Moderate memory usage detected",
                "- Consider periodic cache cleanup",
                "- Monitor for memory leaks"
            ])
        else:
            suggestions.append("Memory usage is optimal")
        
        return suggestions

class CPUOptimizer:
    """CPU usage optimization utilities"""
    
    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """Get detailed CPU information"""
        return {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'current_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'usage_percent': psutil.cpu_percent(interval=1),
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        }
    
    @staticmethod
    def suggest_cpu_optimizations(cpu_usage: float, core_count: int) -> List[str]:
        """Suggest CPU optimization strategies"""
        suggestions = []
        
        if cpu_usage > 90:
            suggestions.extend([
                "Critical: CPU usage is very high",
                "- Reduce batch processing size",
                "- Decrease number of worker threads",
                "- Enable CPU throttling"
            ])
        elif cpu_usage > 70:
            suggestions.extend([
                "Warning: CPU usage is high",
                "- Consider reducing concurrent operations",
                "- Optimize algorithm complexity"
            ])
        
        if core_count <= 2:
            suggestions.extend([
                "Low-core system detected",
                "- Disable parallel processing",
                "- Use sequential operations",
                "- Reduce thread pool size"
            ])
        
        return suggestions

# Usage example and testing
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = PerformanceOptimizer()
    
    # Print system information
    optimizer.print_system_info()
    
    # Get optimized configuration
    config = optimizer.get_optimized_config()
    
    print("\nOPTIMIZED CONFIGURATION:")
    print("-" * 30)
    for key, value in config.items():
        if key != 'detected_hardware':
            print(f"{key}: {value}")
    
    # Save configuration
    optimizer.save_config()
    
    # Memory optimization example
    memory_info = MemoryOptimizer.get_memory_usage()
    print(f"\nCURRENT MEMORY USAGE:")
    print(f"RSS: {memory_info['rss_mb']:.1f} MB")
    print(f"Percent: {memory_info['percent']:.1f}%")
    
    suggestions = MemoryOptimizer.suggest_memory_optimizations(
        memory_info['rss_mb'], 
        config['max_memory_usage']
    )
    print("\nMEMORY OPTIMIZATION SUGGESTIONS:")
    for suggestion in suggestions:
        print(f"  {suggestion}")
    
    # CPU optimization example
    cpu_info = CPUOptimizer.get_cpu_info()
    print(f"\nCURRENT CPU INFO:")
    print(f"Physical cores: {cpu_info['physical_cores']}")
    print(f"Usage: {cpu_info['usage_percent']:.1f}%")
    
    cpu_suggestions = CPUOptimizer.suggest_cpu_optimizations(
        cpu_info['usage_percent'],
        cpu_info['physical_cores']
    )
    print("\nCPU OPTIMIZATION SUGGESTIONS:")
    for suggestion in cpu_suggestions:
        print(f"  {suggestion}")