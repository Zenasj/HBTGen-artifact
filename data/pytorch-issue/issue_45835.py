import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lltm_cuda',
    ext_modules=[
        CUDAExtension(
            'lltm_cuda',
            ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'],
            extra_compile_args={
                'cxx': [],
                # 'nvcc': [],
            }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)