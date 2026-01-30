import torch

from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

os.environ['CXX'] = '/usr/bin/clang++'

setup(name='sparse_op',
      ext_modules=[
          cpp_extension.CppExtension(
              'sparse_op',
              [
                  'src/sparse_op.cpp'
              ],
              extra_compile_args=['-stdlib=libc++'],
              extra_link_args=['-stdlib=libc++']
          )
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension}, install_requires=['torch'])