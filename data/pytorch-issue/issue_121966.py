# torch.rand(50001, 3072, dtype=torch.bfloat16, device='cuda')
import torch
import torch.nn.functional as F

class MyModel(torch.nn.Module):
    def forward(self, x):
        return F.dropout(x * 5, 0.5, training=True)  # Matches original function's behavior

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(50001, 3072, dtype=torch.bfloat16, device='cuda')

