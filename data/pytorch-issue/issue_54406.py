import torch

setup.py
from setuptools import setup

from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
)

ext_modules_cpp = [
    CppExtension(name="name.custom_cpp", sources=["custom_cpp.cpp"]),
]

setup(
    name="name",
    version="0.0.1",
    ext_modules=ext_modules_cpp,
    cmdclass={"build_ext": BuildExtension},
    extras_require={"test": "pytest"},
    zip_safe=False,
)