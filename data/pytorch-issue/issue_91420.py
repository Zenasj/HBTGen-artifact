import torch.nn as nn

import torch

v0 = torch.scalar_tensor(True, dtype=torch.float)
v3 = v0.expand([1, 1, 1])
torch.sort(v3, stable=True, dim=1, descending=True) # It is fine if we don't use `stable=True`.

"""
Collecting environment information...
PyTorch version: 1.14.0.dev20221202+cu117
Is debug build: False
CUDA used to build PyTorch: 11.7
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.1 LTS (x86_64)
GCC version: (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0
Clang version: Could not collect
CMake version: version 3.25.0
Libc version: glibc-2.35

Python version: 3.9.12 (main, Apr  5 2022, 06:56:58)  [GCC 7.5.0] (64-bit runtime)
Python platform: Linux-5.15.0-56-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 11.6.124
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA GeForce RTX 3090
GPU 1: NVIDIA GeForce RTX 3090
GPU 2: NVIDIA GeForce RTX 3090

Nvidia driver version: 515.86.01
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.4.1
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.4.1
/usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.4.1
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8.4.1
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_train.so.8.4.1
/usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8.4.1
/usr/lib/x86_64-linux-gnu/libcudnn_ops_train.so.8.4.1
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

Versions of relevant libraries:
[pip3] mypy-extensions==0.4.3
[pip3] numpy==1.23.3
[pip3] onnx2torch==1.5.3
[pip3] torch==1.14.0.dev20221202+cu117
[pip3] torchaudio==0.14.0.dev20221203+cu117
[pip3] torchtriton==2.0.0+0d7e753227
[pip3] torchvision==0.15.0.dev20221203+cpu
[conda] numpy                     1.23.3                   pypi_0    pypi
[conda] onnx2torch                1.5.3                    pypi_0    pypi
[conda] torch                     1.14.0.dev20221202+cu117          pypi_0    pypi
[conda] torchaudio                0.14.0.dev20221203+cu117          pypi_0    pypi
[conda] torchtriton               2.0.0+0d7e753227          pypi_0    pypi
[conda] torchvision               0.15.0.dev20221203+cpu          pypi_0    pypi
"""

import torch

torch.sort(torch.Tensor([[[True]]]), stable=True, dim=1, descending=True)