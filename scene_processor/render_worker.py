import os
import subprocess
import time
from typing import Optional, Dict
import psutil

def check_gpu_memory() -> Optional[float]:
    """Check GPU memory usage (GB).
    
    Returns:
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 ** 3)  # GB
    except Exception:
        return None

def wait_for_gpu_memory(
    target_free_gb: float = 2.0,
    max_wait_seconds: int = 300,
    check_interval: float = 1.0
) -> bool:
    """Wait for GPU memory to be released enough.
    
    Args:
        target_free_gb:
        max_wait_seconds:
        check_interval:
        
    Returns:
    """
    start_time = time.time()
    
    while time.time() - start_time < max_wait_seconds:
        gpu_memory = check_gpu_memory()
        
        if gpu_memory is None:

            return True
        
        if gpu_memory < target_free_gb:
            return True
        
        time.sleep(check_interval)
    
    return False

def kill_blender_processes():
    """Kill running Blender processes."""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'blender' in proc.info['name'].lower():
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

def cleanup_blender_temp_files(temp_dir: Optional[str] = None):
    """Clean up Blender temporary files.
    
    Args:
        temp_dir:
    """
    if temp_dir is None:
        import tempfile
        temp_dir = tempfile.gettempdir()
    
    import glob
    patterns = [
        os.path.join(temp_dir, "blender_*.exr"),
        os.path.join(temp_dir, "*.blend1"),
    ]
    
    for pattern in patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
            except Exception:
                pass
