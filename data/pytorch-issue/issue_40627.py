import sys
import torch.cuda
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME

ext_modules = [
    CppExtension(
        'test_module',
        ['src/pybind/test_pybindings.cpp'],
        extra_compile_args=['-O3', '-g', '-Werror', '-fopenmp'])
]
setup(name='test_module', ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension})

import torch
import torch.nn
from torch.nn import Module
import test_module

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='test_net',
    ext_modules=[
        CppExtension('test_net', [
            'src/pybind/test_net.cpp'
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)