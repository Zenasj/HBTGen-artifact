import torch.nn as nn

import torch

import os, sys, ctypes
def _preload_cuda_deps():
    """Preloads cudnn/cublas deps if they could not be found otherwise."""
    cublas_path = None
    cudnn_path = None
    for path in sys.path:
        nvidia_path = os.path.join(path, 'nvidia')
        if not os.path.exists(nvidia_path):
            continue
        candidate_cublas_path = os.path.join(nvidia_path, 'cublas', 'lib', 'libcublas.so.11')
        if os.path.exists(candidate_cublas_path) and not cublas_path:
            cublas_path = candidate_cublas_path
        candidate_cudnn_path = os.path.join(nvidia_path, 'cudnn', 'lib', 'libcudnn.so.8')
        if os.path.exists(candidate_cudnn_path) and not cudnn_path:
            cudnn_path = candidate_cudnn_path
        if cublas_path and cudnn_path:
            break
    if not cublas_path or not cudnn_path:
        raise ValueError(f"cublas and cudnn not found in the system path {sys.path}")

    ctypes.CDLL(cublas_path)
    ctypes.CDLL(cudnn_path)


_preload_cuda_deps()
import torch