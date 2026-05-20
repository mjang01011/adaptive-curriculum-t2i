"""GPU memory and utilization stats via torch.cuda + optional pynvml."""
from typing import Dict


def get_gpu_stats(device_index: int = 0) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    try:
        import torch
        if not torch.cuda.is_available():
            return stats
        idx = device_index
        stats["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated(idx) / 1e9
        stats["gpu/memory_reserved_gb"] = torch.cuda.memory_reserved(idx) / 1e9
        total = torch.cuda.get_device_properties(idx).total_memory
        stats["gpu/memory_total_gb"] = total / 1e9
        stats["gpu/memory_used_pct"] = torch.cuda.memory_allocated(idx) / total * 100
    except Exception:
        pass

    # GPU utilization % via pynvml (optional)
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        stats["gpu/utilization_pct"] = float(util.gpu)
        stats["gpu/memory_utilization_pct"] = float(util.memory)
    except Exception:
        pass

    return stats
