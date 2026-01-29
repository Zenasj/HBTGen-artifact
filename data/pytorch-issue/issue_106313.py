# torch.rand(1, 8, 8, dtype=torch.float32, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        U, S, Vh = torch.linalg.svd(x, full_matrices=True)
        permute = Vh.permute(0, 2, 1)
        permute_1 = permute.permute(0, 2, 1)
        expand = U.expand(1, 8, 8)
        view = expand.view(1, 8, 8)
        expand_1 = permute_1.expand(1, 8, 8)
        view_1 = expand_1.view(1, 8, 8)
        bmm_result = torch.bmm(view, view_1)
        view_2 = bmm_result.view(1, 8, 8)
        view_3 = view_2.view(1, 8, 8)
        return view_3

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 8, 8, dtype=torch.float32, device='cuda')

