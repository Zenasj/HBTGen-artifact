from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='diff_cpp',
      ext_modules=[cpp_extension.CppExtension('diff_cpp', ['diff.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

import torch
import diff_cpp    # fine
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = torch.sum(x)
diff_cpp.backw(y)  # hangs