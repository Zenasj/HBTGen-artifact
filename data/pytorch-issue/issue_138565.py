import torch

py
from torch.utils.cpp_extension import load
my_lib = load(name='my_cuda_kernel', sources=['my_cuda_kernel.cu'], extra_cuda_cflags=['-O2', '-std=c++17'])
# ......