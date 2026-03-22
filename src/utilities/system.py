"""
Module for system-level utilities (GPU memory monitoring, etc.).
"""
import gc
import torch

def get_memory_usage_mb():
    """
    Return current GPU memory allocation in megabytes.
    """
    if torch.cuda.is_available():
        # Reserved memory includes allocator cache (more accurate)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        return reserved
    return 0.0

def get_memory_allocated_mb():
    """
    Return only allocated tensor memory (underreports actual usage).
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        return allocated
    return 0.0

def clear_gpu_memory():
    """
    Completely free GPU memory.
    """

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
