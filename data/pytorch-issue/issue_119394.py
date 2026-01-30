from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, include_paths
import torch

import os
from typing import List, Set

ext_modules = []
cpu_extension = CppExtension(
   name='cpu_ops',
   sources=['/library/cpu/cpu_mm.cpp'],
    )
ext_modules.append(cpu_extension)


setup(
    name="kernel_test",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
)