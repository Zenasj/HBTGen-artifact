import torch.nn as nn

import math
import os
import time
from random import Random

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['TORCH_CUDNN_V8_API_DISABLED'] = '1'

import torch

class TestModel(torch.nn.Sequential):
    def __init__(self, in_chans=4, ch64: int = 64):
        super().__init__(
            torch.nn.Conv2d(in_chans, ch64, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(ch64, ch64, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(ch64, in_chans, 3, padding=1),
        )

if __name__ == '__main__':
    print(f'MPx {torch.__version__}\tGpuMs {torch.__version__}')
    use_channels_last = False  # <-- Change this to True if you have a fancy GPU, e.g. A100, A10g, etc
    mf = torch.channels_last if use_channels_last else torch.contiguous_format
    net = TestModel().to(device='cuda', memory_format=mf)
    rnd = Random(42)
    w = h = 512
    while True:
        px = (rnd.random() * 3.9 + 0.1) * 1e6
        aspect_ratio = rnd.random() + 0.5
        w = int(math.sqrt(px * aspect_ratio))
        h = int(px / w)
        x = torch.empty((1, 4, w, h), device='cuda', memory_format=mf)
        t0 = time.time()
        y = net(x).to(device='cpu')
        dt = time.time() - t0
        print(f'{w * h * 1e-6:.3f}\t{dt*1000:.1f}')

import math
import os
import time
from random import Random

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['TORCH_CUDNN_V8_API_DISABLED'] = '1'

import torch


class TestModel(torch.nn.Sequential):
    def __init__(self, in_chans=4, ch64: int = 64):
        super().__init__(
            torch.nn.Conv2d(in_chans, ch64, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(ch64, ch64, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(ch64, in_chans, 3, padding=1),
        )


if __name__ == '__main__':
    dtype = torch.float16
    use_channels_last = False
    mf = torch.channels_last if use_channels_last else torch.contiguous_format
    net = TestModel().to(dtype=dtype, device='cuda', memory_format=mf).eval()
    rnd = Random(42)
    w = h = 1024
    print(f'MPx\tGPU Ms {torch.__version__} {net.__class__.__name__} {dtype} {mf}')
    with torch.no_grad():
        for _ in range(1000):
            px = (rnd.random() * 3.9 + 0.1) * 1e6
            aspect_ratio = rnd.random() + 0.5
            w = int(math.sqrt(px * aspect_ratio))
            h = int(px / w)
            x = torch.empty((1, 4, w, h), dtype=dtype, device='cuda', memory_format=mf)
            t0 = time.time()
            y = net(x)
            torch.cuda.synchronize()
            dt = time.time() - t0
            print(f'{w * h * 1e-6:.3f}\t{dt * 1000:.1f}')

import math
import os
import time
from random import Random

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['TORCH_CUDNN_V8_API_DISABLED'] = '1'

import torch

# Download u2net implementation from:
# https://raw.githubusercontent.com/xuebinqin/U-2-Net/master/model/u2net_refactor.py
from u2net_refactor import U2NET_full

if __name__ == '__main__':
    dtype = torch.float16
    use_channels_last = False
    mf = torch.channels_last if use_channels_last else torch.contiguous_format
    net = U2NET_full().to(dtype=dtype, device='cuda', memory_format=mf).eval()
    rnd = Random(42)
    w = h = 1024
    print(f'MPx\tGPU Ms {torch.__version__} {net.__class__.__name__} {dtype} {mf}')
    with torch.no_grad():
        for _ in range(1000):
            px = (rnd.random() * 1.9 + 0.1) * 1e6
            aspect_ratio = rnd.random() + 0.5
            w = int(math.sqrt(px * aspect_ratio))
            h = int(px / w)
            x = torch.empty((1, 3, w, h), dtype=dtype, device='cuda', memory_format=mf)
            t0 = time.time()
            y = net(x)
            torch.cuda.synchronize()
            dt = time.time() - t0
            print(f'{w * h * 1e-6:.3f}\t{dt * 1000:.1f}')

import math
import os
import time
from random import Random

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
# os.environ['TORCH_CUDNN_V8_API_DISABLED'] = '1'

import torch

# Download u2net implementation from:
# https://raw.githubusercontent.com/xuebinqin/U-2-Net/master/model/u2net_refactor.py
from u2net_refactor import U2NET_full

if __name__ == '__main__':
    dtype = torch.float16
    use_channels_last = False
    mf = torch.channels_last if use_channels_last else torch.contiguous_format
    net = U2NET_full().to(dtype=dtype, device='cuda', memory_format=mf).eval()
    rnd = Random(42)
    w = h = 1024
    print(f'MPx\tGPU Ms {torch.__version__} {net.__class__.__name__} {dtype} {mf}')
    with torch.no_grad():
        for _ in range(1000):
            px = (rnd.random() * 1.9 + 0.1) * 1e6
            aspect_ratio = rnd.random() + 0.5
            w = int(math.sqrt(px * aspect_ratio))
            h = int(px / w)
            x = torch.empty((1, 3, w, h), dtype=dtype, device='cuda', memory_format=mf)
            t0 = time.time()
            y = net(x)
            torch.cuda.synchronize()
            dt = time.time() - t0
            print(f'{w * h * 1e-6:.3f}\t{dt * 1000:.1f}')