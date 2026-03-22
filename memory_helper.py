# memory_helper.py
# Memory and system monitoring utilities
import psutil
import gc


def get_system_stats():
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    return {
        'cpu_percent': cpu_percent,
        'ram_total_gb': memory.total / (1024**3),
        'ram_used_gb': memory.used / (1024**3),
        'ram_available_gb': memory.available / (1024**3),
        'ram_percent': memory.percent,
    }


def print_system_stats(step_name="System"):
    stats = get_system_stats()
    print(f"\n📊 {step_name}: CPU {stats['cpu_percent']:.1f}%  RAM {stats['ram_percent']:.1f}%")
    return stats


def check_memory_pressure():
    return get_system_stats()['ram_percent'] > 85.0


def cleanup_variables(*variables):
    for var in variables:
        if var is not None:
            del var
    gc.collect()
