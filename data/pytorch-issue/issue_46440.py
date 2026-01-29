# Input shape: four tensors of (1, 256), (1, 512), (1, 140), (1, 140) on CUDA
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        input0, input1, input2, input3 = inputs
        # Unsqueeze and concatenate for attention weights
        unsq3 = input3.unsqueeze(1)  # (1, 1, 140)
        unsq2 = input2.unsqueeze(1)  # (1, 1, 140)
        attention_weights_cat = torch.cat((unsq3, unsq2), dim=1)  # (1, 2, 140)
        # Concatenate inputs for cell input
        cell_input = torch.cat((input0, input1), dim=-1)  # (1, 768)
        return cell_input, attention_weights_cat

def my_model_function():
    return MyModel()

def GetInput():
    return (
        torch.randn(1, 256, device='cuda'),
        torch.randn(1, 512, device='cuda'),
        torch.randn(1, 140, device='cuda'),
        torch.randn(1, 140, device='cuda'),
    )

