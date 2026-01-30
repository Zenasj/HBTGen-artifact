import torch

a = torch.rand(..., device='cpu')

# memory alias, not copy observed in intel x86 pcm(https://github.com/intel/pcm)
b = a.to('amdgpu')