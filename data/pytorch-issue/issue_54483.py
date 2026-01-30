import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dim, count = 8):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(count , dim, dim * 4))
        self.w2 = nn.Parameter(torch.randn(count , dim * 4, dim * 4))
        self.w3 = nn.Parameter(torch.randn(count , dim * 4, dim))
        self.act = nn.ELU(inplace = True)

    def forward(self, x):
        hidden1 = self.act(torch.einsum('end,edh->enh', x, self.w1))
        hidden2 = self.act(torch.einsum('end,edh->enh', hidden1, self.w2))
        out = torch.einsum('end,edh->enh', hidden2, self.w3)
        return out