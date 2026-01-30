import torch

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='some_lib',  # used by `pip install`
    version='0.0.1',
    description='',
    cmdclass={
        'build_ext': BuildExtension
    },
    ext_modules=[
        CUDAExtension(
            'some_lib',
            [
                'src/aaa.cu',
            ],
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': ['-O2', '-allow-unsupported-compiler', '-std=c++20']})
    ],
    setup_requires=["pybind11"],
    install_requires=["pybind11"],
    python_requires='>=3.8',
    include_package_data=True,
    zip_safe=False,
)

nvcc_std = os.popen("nvcc -h | grep -- '--std'")
nvcc_std = nvcc_std.read()

nvcc_flags = ['-O2', '-allow-unsupported-compiler']
if nvcc_std.__contains__('c++20'):
    nvcc_flags.append('-std=c++20')

import os

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Make sure that the nvcc executable is available in $PATH variables,
# or find one according to the $CUDA_HOME variable
nvcc_std = os.popen("nvcc -h | grep -- '--std'")
nvcc_std = nvcc_std.read()

nvcc_flags = ['-O2', '-allow-unsupported-compiler']
if nvcc_std.__contains__('c++20'):
    nvcc_flags.append('-std=c++20')

setup(
    name='some_lib',  # used by `pip install`
    version='0.0.1',
    description='',
    cmdclass={
        'build_ext': BuildExtension
    },
    ext_modules=[
        CUDAExtension(
            'some_lib',
            [
                'src/aaa.cu',
            ],
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': nvcc_flags})
    ],
    setup_requires=["pybind11"],
    install_requires=["pybind11"],
    python_requires='>=3.8',
    include_package_data=True,
    zip_safe=False,
)