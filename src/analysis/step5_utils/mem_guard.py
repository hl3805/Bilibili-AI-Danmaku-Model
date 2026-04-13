import psutil

def check_memory(threshold=90):
    return psutil.virtual_memory().percent > threshold

def memory_status():
    mem = psutil.virtual_memory()
    return {
        'percent': mem.percent,
        'available_gb': mem.available / (1024**3),
        'total_gb': mem.total / (1024**3)
    }
