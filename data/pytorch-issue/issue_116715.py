# torch.rand(1, 320, 128, 128, dtype=torch.float32, device='cuda', requires_grad=True, memory_format=torch.channels_last) ← Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.Dropout(p=0.1),
        )

    def forward(self, x):
        out = F.gelu(x)
        out = self.in_layers(out)
        return out

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    model = MyModel()
    model = model.to("cuda").to(memory_format=torch.channels_last)
    return model

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    input_t = torch.randn([1, 320, 128, 128], dtype=torch.float32, device='cuda', requires_grad=True)
    input_t = input_t.to(memory_format=torch.channels_last)
    return input_t

