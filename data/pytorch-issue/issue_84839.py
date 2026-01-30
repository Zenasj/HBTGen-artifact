import torch

from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_utils import get_cycles_per_ms

stream = torch.cuda.Stream()
sleep_duration_ms = 25
sleep_duration_cycles = int(sleep_duration_ms * get_cycles_per_ms())
tensor_numel = 1 * 1024 * 1024
NON_BLOCKING = True
USE_SEPARATE_STREAM = True

cpu_tensor = torch.ones((tensor_numel,))
cpu_tensor = cpu_tensor.pin_memory()

stream_context = (
    torch.cuda.stream(stream) if USE_SEPARATE_STREAM
    else torch.cuda.stream(torch.cuda.current_stream())
)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with stream_context:
        torch.cuda._sleep(sleep_duration_cycles)
        gpu_tensor = cpu_tensor.to(torch.cuda.current_device(), non_blocking=NON_BLOCKING)

print(f"non-blocking={NON_BLOCKING}")
print(f"use separate stream={USE_SEPARATE_STREAM}")
prof.export_chrome_trace(f"trace_{NON_BLOCKING}_{USE_SEPARATE_STREAM}.json")