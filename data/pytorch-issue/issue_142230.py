# torch.rand(4, 12, 1023, 1022, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def forward(self, x):
        softmax = F.softmax(x, dim=-1)
        sum_result = torch.sum(softmax, dim=-1)
        sum_broadcast = torch.broadcast_to(
            sum_result.unsqueeze(-1), [*sum_result.size()[:3], 256]
        )
        sum_exp = torch.exp(sum_broadcast)
        return torch.sum(sum_exp, dim=-1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 12, 1023, 1022, dtype=torch.float32)

