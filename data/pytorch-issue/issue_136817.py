import torch.nn as nn

import torch
import torch.profiler
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Flatten(),
    torch.nn.Linear(64 * 16 * 16, 10),
)
inputs = torch.randn(5, 3, 224, 224)

# Initialize profiler with CPU and CUDA activities
prof = torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    with_stack=True,
)
# Start profiling
prof.start()
output = model(inputs)
prof.stop()
prof.export_chrome_trace("profiling_trace.json")