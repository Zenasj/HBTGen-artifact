import torch

# option-2: JIT compiled extension
from torch.utils.cpp_extension import load
load(name=name,
     sources=sources,
     extra_include_paths=extra_include_paths,
     extra_cflags=cxx_args,
     extra_cuda_cflags=nvcc_args,
     extra_ldflags=extra_ldflags,
     verbose=verbose)