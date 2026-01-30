import torch

from torch.utils.cpp_extension import BuildExtension, CppExtension
try:
    from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME
    assert CUDA_HOME, "No CUDA found"
except (ImportError, OSError, AssertionError) as e:
    CUDAExtension = None
    print("No CUDA was detected, building without CUDA error: {}".format(e))