# torch.rand(32768, dtype=torch.float32)

import torch
import math

tensor_size = 4  # GB
model_size = 16  # GB
precision_size = 4  # bytes per element (float32)
kB = 1024
MB = kB * kB
GB = MB * kB  # 1024^3 bytes

layers_num = model_size // tensor_size
activation_size = math.floor(math.sqrt((tensor_size * GB) / precision_size))

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(layers_num):
            name = f"fc_{i}"
            linear = torch.nn.Linear(activation_size, activation_size)
            setattr(self, name, linear)
    
    def forward(self, x):
        for i in range(layers_num):
            name = f"fc_{i}"
            x = getattr(self, name)(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(activation_size, dtype=torch.float32)

