# torch.rand(1, 3, 224, 224, dtype=torch.float32, device=torch.device('cuda:0'))  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        half_k = x.numel() // 2
        sort_values, _ = x.view(-1).sort(descending=True)
        topk_values, _ = x.view(-1).topk(half_k, sorted=False)
        kthvalue_value, _ = x.view(-1).kthvalue(half_k)
        
        return sort_values[half_k], topk_values[-1], kthvalue_value

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 3, 224, 224, dtype=torch.float32, device=torch.device('cuda:0'))

