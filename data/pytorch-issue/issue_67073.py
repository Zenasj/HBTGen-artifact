import torch

from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

cublas_module = CUDAExtension(
            name='cublas_ext',
            sources=['cublas_ext.cpp']
        )

setup(
    name='cublas_ext_root',
    version='0.1',
    ext_modules=[cublas_module],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    }
)

# setup.py
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

cublas_module = CUDAExtension(
            name='cuda_ext',
            sources=['cuda_ext.cpp']
        )

setup(
    name='cuda_ext_root',
    version='0.1',
    ext_modules=[cublas_module],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    }
)