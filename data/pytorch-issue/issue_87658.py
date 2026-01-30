import torch
n = 32
val = -0.002336116973310709
val*1/(n-1) == val*(1/(n-1)) # true
torch.Tensor([val])*1/(n-1) == torch.Tensor([val])*(1/(n-1)) # tensor([False])