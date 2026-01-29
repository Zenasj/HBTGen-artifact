# torch.rand(1, 32, 32, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        self.num_experts = 32
        self.top_k = 4
        routing_weights, selected_experts = torch.topk(x, self.top_k, dim=-1)
        one_hot_encoded = torch.nn.functional.one_hot(selected_experts, self.num_experts)  # .permute(2, 1, 0)
        return one_hot_encoded.sum()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn((1, 32, 32))

