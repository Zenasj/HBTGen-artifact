import torch

import os.path as osp

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))

s = "C_extension"
include_dirs = [osp.join(ROOT_DIR)]
sources = ["test.cpp"]
CPU_VERSION = CppExtension(
name=s,
sources=sources,
include_dirs=include_dirs,
extra_compile_args={"cxx": ["-O3", "-std=c++17"]}
)


setup(
    name=s,
    version='0.1',
    author='zbwu',
    author_email='zbwu1996@gmail.com',
    description=s,
    long_description=s,
    ext_modules=[
        CPU_VERSION
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    }
)