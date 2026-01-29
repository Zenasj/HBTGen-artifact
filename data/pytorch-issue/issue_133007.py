import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.rand(1, dtype=torch.float32)  # Dummy input to determine device
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Inferred parameters based on typical transformer layer dimensions
        self.add_param0 = nn.Parameter(torch.randn(1024, 768))  # Shape from __add__ operation
        self.add_param1 = nn.Parameter(torch.randn(1024, 768))
        self.norm_weight = nn.Parameter(torch.randn(768))  # LayerNorm parameters
        self.norm_bias = nn.Parameter(torch.randn(768))
        self.norm_shape = (768,)
        self.norm_eps = 1e-5  # From layer_norm.pt parameter:4
        self.linear_weight = nn.Parameter(torch.randn(512, 768))  # Linear layer parameters
        self.linear_bias = nn.Parameter(torch.randn(512))

    def forward(self, x):
        current_device = x.device
        # Compute on current device
        output = self.add_param0 + self.add_param1
        output = F.layer_norm(output, self.norm_shape, self.norm_weight, self.norm_bias, self.norm_eps)
        output = F.linear(output, self.linear_weight, self.linear_bias)

        # Compute on the other device (CPU/GPU toggle)
        other_device = 'cpu' if current_device.type == 'cuda' else 'cuda'
        other_add0 = self.add_param0.to(other_device)
        other_add1 = self.add_param1.to(other_device)
        other_norm_w = self.norm_weight.to(other_device)
        other_norm_b = self.norm_bias.to(other_device)
        other_linear_w = self.linear_weight.to(other_device)
        other_linear_b = self.linear_bias.to(other_device)
        other_output = other_add0 + other_add1
        other_output = F.layer_norm(other_output, self.norm_shape, other_norm_w, other_norm_b, self.norm_eps)
        other_output = F.linear(other_output, other_linear_w, other_linear_b)

        # Calculate maximum difference between outputs
        max_diff = torch.max(torch.abs(output - other_output.to(current_device)))
        return max_diff

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)  # Dummy input to trigger device selection

