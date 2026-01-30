import torch

import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

setup(name='exten', ext_modules=[
    CUDAExtension('exten_cuda', ['exten_cuda.cpp', 'exten_cuda_kernel.cu'])
    ], cmdclass={'build_ext': BuildExtension})