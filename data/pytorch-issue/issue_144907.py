import torch
import torch.nn as nn

class M(torch.nn.Module):
    def forward(self, h, w, strides):
        stride_tensor = []
        for stride in strides:
            res = torch.full((h * w, 1), stride)
            stride_tensor.append(res)
        
        return torch.cat(stride_tensor)

ep = torch.export.export(M(), (80, 80, torch.tensor([8.0, 16.0, 32.0])), strict=True)
print(ep)