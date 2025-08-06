#!/usr/bin/env python3
"""
Performance Benchmark Suite for Discord AI Chatbot
Tests various components and provides optimization recommendations
"""

import time
import asyncio
import statistics
import json
import os
import sys
from typing import Dict, List, Any, Tuple
import logging
import psutil
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import our optimized modules
from performance_config import PerformanceOptimizer, MemoryOptimizer, CPUOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkSuite:
    """Comprehensive benchmark suite for the Discord AI chatbot"""
    
    def __init__(self):
        self.results = {}
        self.optimizer = PerformanceOptimizer()
        self.start_time = time.time()
        
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark tests"""
        logger.info("Starting comprehensive benchmark suite...")
        
        # System information
        self.results['system_info'] = self.optimizer.system_info
        self.results['timestamp'] = datetime.now().isoformat()
        
        # Run individual benchmarks
        self.results['memory_benchmark'] = self.benchmark_memory_usage()
        self.results['cpu_benchmark'] = self.benchmark_cpu_performance()
        self.results['text_processing_benchmark'] = self.benchmark_text_processing()
        self.results['embedding_benchmark'] = self.benchmark_embeddings()
        self.results['cache_benchmark'] = self.benchmark_caching()
        self.results['concurrent_benchmark'] = self.benchmark_concurrency()
        
        # Generate recommendations
        self.results['recommendations'] = self.generate_recommendations()
        
        # Calculate overall score
        self.results['overall_score'] = self.calculate_overall_score()
        
        total_time = time.time() - self.start_time
        self.results['total_benchmark_time'] = total_time
        
        logger.info(f"Benchmark suite completed in {total_time:.2f} seconds")
        return self.results
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns"""
        logger.info("Running memory usage benchmark...")
        
        # Get baseline memory
        baseline_memory = MemoryOptimizer.get_memory_usage()
        
        # Test memory allocation patterns
        memory_tests = []
        
        # Test 1: Large list allocation
        start_time = time.time()
        large_list = [i for i in range(100000)]
        allocation_time = time.time() - start_time
        current_memory = MemoryOptimizer.get_memory_usage()
        memory_tests.append({
            'test': 'large_list_allocation',
            'time': allocation_time,
            'memory_increase': current_memory['rss_mb'] - baseline_memory['rss_mb']
        })
        
        # Test 2: Dictionary operations
        start_time = time.time()
        large_dict = {str(i): i for i in range(50000)}
        dict_time = time.time() - start_time
        dict_memory = MemoryOptimizer.get_memory_usage()
        memory_tests.append({
            'test': 'dictionary_allocation',
            'time': dict_time,
            'memory_increase': dict_memory['rss_mb'] - current_memory['rss_mb']
        })
        
        # Test 3: Memory cleanup
        start_time = time.time()
        del large_list, large_dict
        import gc
        gc.collect()
        cleanup_time = time.time() - start_time
        final_memory = MemoryOptimizer.get_memory_usage()
        
        return {
            'baseline_memory_mb': baseline_memory['rss_mb'],
            'final_memory_mb': final_memory['rss_mb'],
            'memory_tests': memory_tests,
            'cleanup_time': cleanup_time,
            'memory_recovered_mb': dict_memory['rss_mb'] - final_memory['rss_mb']
        }
    
    def benchmark_cpu_performance(self) -> Dict[str, Any]:
        """Benchmark CPU performance"""
        logger.info("Running CPU performance benchmark...")
        
        cpu_tests = []
        
        # Test 1: Mathematical operations
        start_time = time.time()
        result = sum(i * i for i in range(100000))
        math_time = time.time() - start_time
        cpu_tests.append({
            'test': 'mathematical_operations',
            'time': math_time,
            'operations_per_second': 100000 / math_time
        })
        
        # Test 2: String operations
        start_time = time.time()
        text = "test string " * 1000
        for _ in range(1000):
            text.upper().lower().strip()
        string_time = time.time() - start_time
        cpu_tests.append({
            'test': 'string_operations',
            'time': string_time,
            'operations_per_second': 1000 / string_time
        })
        
        # Test 3: List comprehensions
        start_time = time.time()
        result = [x**2 for x in range(10000) if x % 2 == 0]
        list_comp_time = time.time() - start_time
        cpu_tests.append({
            'test': 'list_comprehensions',
            'time': list_comp_time,
            'operations_per_second': 10000 / list_comp_time
        })
        
        return {
            'cpu_info': CPUOptimizer.get_cpu_info(),
            'cpu_tests': cpu_tests,
            'average_performance': statistics.mean([test['operations_per_second'] for test in cpu_tests])
        }
    
    def benchmark_text_processing(self) -> Dict[str, Any]:
        """Benchmark text processing operations"""
        logger.info("Running text processing benchmark...")
        
        # Sample texts for testing
        sample_texts = [
            "Hello **world**! This is a *test* message with `code` and ~~strikethrough~~.",
            "Check out this link: https://example.com and mention <@123456789>",
            "🎉 Emoji test with [joy] and [fire] emojis! 🔥",
            "Multiple **bold** and *italic* and __underline__ formatting tests.",
            "```python\nprint('code block test')\n```"
        ] * 100  # Multiply for more substantial testing
        
        # Test markdown cleaning
        start_time = time.time()
        cleaned_texts = []
        for text in sample_texts:
            # Simulate the clean_markdown function
            import re
            cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
            cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)
            cleaned = re.sub(r'__(.*?)__', r'\1', cleaned)
            cleaned = re.sub(r'~~(.*?)~~', r'\1', cleaned)
            cleaned = re.sub(r'`(.*?)`', r'\1', cleaned)
            cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL)
            cleaned = re.sub(r'<@!?(\d+)>', '', cleaned)
            cleaned = re.sub(r'https?://\S+', '', cleaned)
            cleaned_texts.append(cleaned.strip())
        
        markdown_time = time.time() - start_time
        
        # Test emoji processing
        start_time = time.time()
        emoji_texts = []
        for text in sample_texts:
            # Simulate emoji processing
            processed = text.replace('🎉', '[party]').replace('🔥', '[fire]')
            emoji_texts.append(processed)
        
        emoji_time = time.time() - start_time
        
        return {
            'markdown_cleaning_time': markdown_time,
            'emoji_processing_time': emoji_time,
            'texts_processed': len(sample_texts),
            'markdown_speed_texts_per_second': len(sample_texts) / markdown_time,
            'emoji_speed_texts_per_second': len(sample_texts) / emoji_time
        }
    
    def benchmark_embeddings(self) -> Dict[str, Any]:
        """Benchmark embedding generation"""
        logger.info("Running embedding benchmark...")
        
        # Sample texts for embedding
        sample_texts = [
            "This is a test message for embedding generation.",
            "Another sample text with different content and meaning.",
            "Machine learning and artificial intelligence are fascinating topics.",
            "Discord bots can be very useful for community management.",
            "Performance optimization is crucial for scalable applications."
        ] * 20  # 100 total texts
        
        embedding_tests = []
        
        # Test single embedding generation
        single_text = sample_texts[0]
        start_time = time.time()
        
        # Simulate embedding generation (using numpy for speed)
        single_embedding = np.random.rand(384)  # Simulate 384-dim embedding
        single_time = time.time() - start_time
        
        embedding_tests.append({
            'test': 'single_embedding',
            'time': single_time,
            'dimension': 384
        })
        
        # Test batch embedding generation
        batch_size = 32
        batches = [sample_texts[i:i+batch_size] for i in range(0, len(sample_texts), batch_size)]
        
        start_time = time.time()
        batch_embeddings = []
        for batch in batches:
            # Simulate batch embedding generation
            batch_emb = np.random.rand(len(batch), 384)
            batch_embeddings.append(batch_emb)
        
        batch_time = time.time() - start_time
        
        embedding_tests.append({
            'test': 'batch_embedding',
            'time': batch_time,
            'total_texts': len(sample_texts),
            'batch_size': batch_size,
            'texts_per_second': len(sample_texts) / batch_time
        })
        
        return {
            'embedding_tests': embedding_tests,
            'single_embedding_time': single_time,
            'batch_embedding_time': batch_time,
            'batch_speedup': single_time * len(sample_texts) / batch_time
        }
    
    def benchmark_caching(self) -> Dict[str, Any]:
        """Benchmark caching performance"""
        logger.info("Running caching benchmark...")
        
        # Simulate cache operations
        cache_data = {}
        cache_tests = []
        
        # Test cache writes
        start_time = time.time()
        for i in range(1000):
            cache_data[f"key_{i}"] = f"value_{i}" * 10
        write_time = time.time() - start_time
        
        cache_tests.append({
            'test': 'cache_writes',
            'time': write_time,
            'operations': 1000,
            'ops_per_second': 1000 / write_time
        })
        
        # Test cache reads
        start_time = time.time()
        for i in range(1000):
            value = cache_data.get(f"key_{i}")
        read_time = time.time() - start_time
        
        cache_tests.append({
            'test': 'cache_reads',
            'time': read_time,
            'operations': 1000,
            'ops_per_second': 1000 / read_time
        })
        
        # Test cache misses
        start_time = time.time()
        for i in range(1000, 2000):
            value = cache_data.get(f"key_{i}", "default")
        miss_time = time.time() - start_time
        
        cache_tests.append({
            'test': 'cache_misses',
            'time': miss_time,
            'operations': 1000,
            'ops_per_second': 1000 / miss_time
        })
        
        return {
            'cache_tests': cache_tests,
            'cache_size': len(cache_data),
            'read_write_ratio': read_time / write_time
        }
    
    def benchmark_concurrency(self) -> Dict[str, Any]:
        """Benchmark concurrent operations"""
        logger.info("Running concurrency benchmark...")
        
        async def async_task(task_id: int, duration: float = 0.01):
            """Simulate an async task"""
            await asyncio.sleep(duration)
            return f"Task {task_id} completed"
        
        def sync_task(task_id: int, duration: float = 0.01):
            """Simulate a sync task"""
            time.sleep(duration)
            return f"Task {task_id} completed"
        
        concurrency_tests = []
        
        # Test async concurrency
        async def test_async_concurrency():
            start_time = time.time()
            tasks = [async_task(i) for i in range(100)]
            results = await asyncio.gather(*tasks)
            return time.time() - start_time
        
        # Run async test
        async_time = asyncio.run(test_async_concurrency())
        concurrency_tests.append({
            'test': 'async_concurrency',
            'time': async_time,
            'tasks': 100,
            'tasks_per_second': 100 / async_time
        })
        
        # Test sync sequential
        start_time = time.time()
        sync_results = [sync_task(i) for i in range(100)]
        sync_time = time.time() - start_time
        
        concurrency_tests.append({
            'test': 'sync_sequential',
            'time': sync_time,
            'tasks': 100,
            'tasks_per_second': 100 / sync_time
        })
        
        return {
            'concurrency_tests': concurrency_tests,
            'async_speedup': sync_time / async_time,
            'recommended_concurrency': min(100, psutil.cpu_count() * 2)
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on benchmark results"""
        recommendations = []
        
        # Memory recommendations
        memory_result = self.results.get('memory_benchmark', {})
        if memory_result.get('final_memory_mb', 0) > 1000:
            recommendations.append("High memory usage detected. Consider reducing cache sizes.")
        
        # CPU recommendations
        cpu_result = self.results.get('cpu_benchmark', {})
        avg_performance = cpu_result.get('average_performance', 0)
        if avg_performance < 10000:
            recommendations.append("Low CPU performance. Consider using lighter algorithms.")
        
        # Text processing recommendations
        text_result = self.results.get('text_processing_benchmark', {})
        if text_result.get('markdown_speed_texts_per_second', 0) < 1000:
            recommendations.append("Slow text processing. Consider caching cleaned text.")
        
        # Embedding recommendations
        embedding_result = self.results.get('embedding_benchmark', {})
        if embedding_result.get('batch_speedup', 1) > 5:
            recommendations.append("Batch processing shows significant speedup. Enable batch mode.")
        
        # Concurrency recommendations
        concurrency_result = self.results.get('concurrent_benchmark', {})
        if concurrency_result.get('async_speedup', 1) > 2:
            recommendations.append("Async operations show good speedup. Increase async usage.")
        
        if not recommendations:
            recommendations.append("Performance looks good! No major optimizations needed.")
        
        return recommendations
    
    def calculate_overall_score(self) -> float:
        """Calculate overall performance score (0-100)"""
        scores = []
        
        # Memory score (lower usage is better)
        memory_result = self.results.get('memory_benchmark', {})
        memory_mb = memory_result.get('final_memory_mb', 500)
        memory_score = max(0, 100 - (memory_mb / 10))  # 1000MB = 0 points
        scores.append(memory_score)
        
        # CPU score (higher performance is better)
        cpu_result = self.results.get('cpu_benchmark', {})
        avg_performance = cpu_result.get('average_performance', 10000)
        cpu_score = min(100, avg_performance / 1000)  # 100k ops/sec = 100 points
        scores.append(cpu_score)
        
        # Text processing score
        text_result = self.results.get('text_processing_benchmark', {})
        text_speed = text_result.get('markdown_speed_texts_per_second', 1000)
        text_score = min(100, text_speed / 50)  # 5000 texts/sec = 100 points
        scores.append(text_score)
        
        # Concurrency score
        concurrency_result = self.results.get('concurrent_benchmark', {})
        async_speedup = concurrency_result.get('async_speedup', 1)
        concurrency_score = min(100, async_speedup * 20)  # 5x speedup = 100 points
        scores.append(concurrency_score)
        
        return statistics.mean(scores)
    
    def save_results(self, filename: str = None):
        """Save benchmark results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"Benchmark results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("DISCORD AI CHATBOT PERFORMANCE BENCHMARK RESULTS")
        print("="*60)
        
        # System info
        system_info = self.results['system_info']
        print(f"\nSYSTEM INFORMATION:")
        print(f"  RAM: {system_info['ram_gb']:.1f} GB")
        print(f"  CPU Cores: {system_info['cpu_cores']}")
        print(f"  Platform: {system_info['platform']}")
        
        # Overall score
        print(f"\nOVERALL PERFORMANCE SCORE: {self.results['overall_score']:.1f}/100")
        
        # Individual benchmark results
        print(f"\nBENCHMARK RESULTS:")
        
        # Memory
        memory_result = self.results['memory_benchmark']
        print(f"  Memory Usage: {memory_result['final_memory_mb']:.1f} MB")
        print(f"  Memory Recovered: {memory_result['memory_recovered_mb']:.1f} MB")
        
        # CPU
        cpu_result = self.results['cpu_benchmark']
        print(f"  CPU Performance: {cpu_result['average_performance']:.0f} ops/sec")
        
        # Text Processing
        text_result = self.results['text_processing_benchmark']
        print(f"  Text Processing: {text_result['markdown_speed_texts_per_second']:.0f} texts/sec")
        
        # Embeddings
        embedding_result = self.results['embedding_benchmark']
        print(f"  Embedding Speed: {embedding_result['embedding_tests'][1]['texts_per_second']:.0f} texts/sec")
        print(f"  Batch Speedup: {embedding_result['batch_speedup']:.1f}x")
        
        # Concurrency
        concurrency_result = self.results['concurrent_benchmark']
        print(f"  Async Speedup: {concurrency_result['async_speedup']:.1f}x")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        for i, rec in enumerate(self.results['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print(f"\nTotal Benchmark Time: {self.results['total_benchmark_time']:.2f} seconds")
        print("="*60)

def create_performance_chart(results: Dict[str, Any]):
    """Create performance visualization charts"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Discord AI Chatbot Performance Benchmark', fontsize=16)
        
        # Memory usage chart
        memory_data = results['memory_benchmark']['memory_tests']
        memory_names = [test['test'] for test in memory_data]
        memory_times = [test['time'] for test in memory_data]
        
        axes[0, 0].bar(memory_names, memory_times)
        axes[0, 0].set_title('Memory Operations Time')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # CPU performance chart
        cpu_data = results['cpu_benchmark']['cpu_tests']
        cpu_names = [test['test'] for test in cpu_data]
        cpu_ops = [test['operations_per_second'] for test in cpu_data]
        
        axes[0, 1].bar(cpu_names, cpu_ops)
        axes[0, 1].set_title('CPU Performance')
        axes[0, 1].set_ylabel('Operations per Second')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Text processing comparison
        text_data = results['text_processing_benchmark']
        text_categories = ['Markdown Cleaning', 'Emoji Processing']
        text_speeds = [
            text_data['markdown_speed_texts_per_second'],
            text_data['emoji_speed_texts_per_second']
        ]
        
        axes[1, 0].bar(text_categories, text_speeds)
        axes[1, 0].set_title('Text Processing Speed')
        axes[1, 0].set_ylabel('Texts per Second')
        
        # Overall score gauge
        score = results['overall_score']
        colors = ['red' if score < 50 else 'yellow' if score < 75 else 'green']
        axes[1, 1].pie([score, 100-score], labels=['Score', 'Remaining'], 
                       colors=[colors[0], 'lightgray'], startangle=90)
        axes[1, 1].set_title(f'Overall Score: {score:.1f}/100')
        
        plt.tight_layout()
        
        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_filename = f"performance_chart_{timestamp}.png"
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Performance chart saved as {chart_filename}")
        
    except ImportError:
        logger.warning("Matplotlib not available. Skipping chart generation.")
    except Exception as e:
        logger.error(f"Error creating performance chart: {e}")

def main():
    """Main benchmark execution"""
    print("Discord AI Chatbot Performance Benchmark Suite")
    print("=" * 50)
    
    # Initialize and run benchmarks
    benchmark = BenchmarkSuite()
    results = benchmark.run_all_benchmarks()
    
    # Print summary
    benchmark.print_summary()
    
    # Save results
    benchmark.save_results()
    
    # Create performance chart
    create_performance_chart(results)
    
    # Generate performance config
    optimizer = PerformanceOptimizer()
    config = optimizer.get_optimized_config()
    
    print(f"\nOptimized configuration generated based on your hardware:")
    print(f"Recommended profile: {config['recommended_profile']}")
    print(f"Max memory usage: {config['max_memory_usage']} MB")
    print(f"Max workers: {config['max_workers']}")
    print(f"Batch size: {config['batch_size']}")

if __name__ == "__main__":
    main()