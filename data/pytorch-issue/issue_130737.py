import torch

import os
from torch.utils.cpp_extension import load

sources = [
    os.path.join(os.path.dirname(__file__), "csrc", "cuda_impl.cpp"),
    os.path.join(os.path.dirname(__file__), "csrc", "cuda_kernel.cu"),
]
load(
    name="my_cuda",
    sources=sources,
    extra_cflags=['-O3'],
    is_python_module=False,
    verbose=False
)
print("OK")