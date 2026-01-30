from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='f',
      ext_modules=[CUDAExtension('f', ['kernel.cpp', 'kernel_cuda.cu'])],
      cmdclass={'build_ext': BuildExtension})

import torch
import f
a = torch.ones(256).cuda()
b = torch.ones(256).cuda()
c = torch.zeros(256).cuda()
f.f_cpp(a, b, c)
print(c)