import torch
import torch.nn as nn
import torch.nn.functional as F

Python
from torch.nn.attention import sdpa_kernel, SDPBackend    

with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
    out = F.scaled_dot_product_attention(q, k, v)