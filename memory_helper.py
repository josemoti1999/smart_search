# memory_helper.py
# Memory and system monitoring utilities
import psutil  # For CPU and RAM monitoring
import gc  # For garbage collection
import torch

def get_system_stats():
    """Get current system CPU and RAM statistics."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    stats = {
        'cpu_percent': cpu_percent,
        'ram_total_gb': memory.total / (1024**3),
        'ram_used_gb': memory.used / (1024**3),
        'ram_available_gb': memory.available / (1024**3),
        'ram_percent': memory.percent
    }
    return stats

def print_system_stats(step_name="System"):
    """Print current system statistics in a formatted way."""
    stats = get_system_stats()
    print(f"\n📊 {step_name} Resources:")
    print(f"   CPU Usage: {stats['cpu_percent']:.1f}%")
    print(f"   RAM: {stats['ram_used_gb']:.2f}GB / {stats['ram_total_gb']:.2f}GB ({stats['ram_percent']:.1f}%)")
    print(f"   Available RAM: {stats['ram_available_gb']:.2f}GB")
    return stats

def check_memory_pressure():
    """Check if system is under memory pressure."""
    stats = get_system_stats()
    return stats['ram_percent'] > 85.0  # Return True if RAM usage > 85%

def cleanup_variables(*variables):
    """Clean up specified variables and run garbage collection."""
    for var in variables:
        if var is not None:
            del var
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
