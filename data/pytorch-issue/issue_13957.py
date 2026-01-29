# torch.rand(256, 256, dtype=torch.float, requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, X):
        S = X.size(0)
        Z = X.sum(1, keepdim=True) + torch.linspace(-S, S, S, device=X.device)
        Y = X * torch.erf(Z).sum(1, keepdim=True)
        mat = Y.t().mm(Y)
        inv_mat = torch.inverse(mat)
        loss = inv_mat.sum() + Y.sum()
        return loss

def my_model_function():
    return MyModel()

def GetInput():
    S = 256
    return torch.randn(S, S, dtype=torch.float, requires_grad=True)

