import torch

def forward(x, device):

  x = torch.trunc(out=x, input=torch.rand([9, 10], dtype=torch.float32).to('cpu'))