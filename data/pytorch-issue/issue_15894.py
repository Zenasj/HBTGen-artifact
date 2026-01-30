####################
## build.py
####################
import os.path as osp

import torch
from torch.utils.ffi import create_extension

sources = []
headers = []
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/sync_bn.cpp']
    headers += ['src/sync_bn.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

assert with_cuda

current_dir = osp.dirname(osp.realpath(__file__))
extra_objects = ['src/cuda/sync_bn_kernel.o']
extra_objects = [osp.join(current_dir, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.sync_bn_lib',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    extra_compile_args=['/MT'],
    libraries=['caffe2_gpu', 'cudart', '_C', 'caffe2'] )

if __name__ == '__main__':
    ffi.build()