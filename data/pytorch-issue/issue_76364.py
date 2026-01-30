import torch

torch.mul(torch.zeros(3, device='cuda'), 2.5) # CUDA Tensor and CPU Scalar