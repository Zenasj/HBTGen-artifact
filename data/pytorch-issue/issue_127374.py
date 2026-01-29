# torch.rand(16, 1, dtype=torch.uint8)  # Input shape (16, 1) with uint8 dtype
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        uint8_data = x.to(torch.uint8)
        rshift = uint8_data >> 6
        first_elements = rshift & 3
        rshift_1 = uint8_data >> 4
        second_elements = rshift_1 & 3
        rshift_2 = uint8_data >> 2
        third_elements = rshift_2 & 3
        fourth_elements = uint8_data & 3
        stacked = torch.stack((first_elements, second_elements, third_elements, fourth_elements), dim=-1)
        return stacked.view(16, 4)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 256, (16, 1), dtype=torch.uint8)

