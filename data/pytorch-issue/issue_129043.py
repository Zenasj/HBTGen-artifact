# torch.rand(B, 1024, dtype=torch.bfloat16, device='cuda'), torch.randint(0, 50257, (B,), device='cuda')
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1024, 50257, dtype=torch.bfloat16).to('cuda')

    def forward(self, inputs):
        x, targets = inputs
        logits = self.linear(x)
        loss = F.cross_entropy(logits, targets)
        return loss

def my_model_function():
    return MyModel()

def GetInput():
    B = 32768
    x = torch.rand(B, 1024, dtype=torch.bfloat16, device='cuda')
    targets = torch.randint(0, 50257, (B,), device='cuda')
    return (x, targets)

