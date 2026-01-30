import torch

print(torch.__version__)
# 2.0.1+cu117

print(torch.cuda.get_device_properties(0))
# /scratch/torchbuild/lib/python3.10/site-packages/torch/cuda/__init__.py:173: UserWarning:
# NVIDIA H100 PCIe with CUDA capability sm_90 is not compatible with the current PyTorch installation.
# The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86.
# If you want to use the NVIDIA H100 PCIe GPU with PyTorch, please check the instructions at https://pytorch.org/get-# started/locally/

# warnings.warn(incompatible_device_warn.format(device_name, capability, " ".join(arch_list), device_name))
# _CudaDeviceProperties(name='NVIDIA H100 PCIe', major=9, minor=0, total_memory=81008MB, multi_processor_count=114)

print(torch.randn(1, device="cuda"))
# Traceback (most recent call last):
# File "<stdin>", line 1, in <module>
# RuntimeError: CUDA error: no kernel image is available for execution on the device
# CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
# For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
# Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.