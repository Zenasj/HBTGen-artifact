# torch.rand(1, 4, 5, 5, dtype=torch.quint8)
import torch
import torch.nn as nn
import torch.nn.quantized.functional as qF

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filters = nn.Parameter(torch.randn(8, 4, 3, 3, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(8, dtype=torch.float))
        self.scale = 1.0
        self.zero_point = 0
        self.padding = 1

    def forward(self, q_inputs):
        q_filters = torch.quantize_per_tensor(self.filters, self.scale, self.zero_point, torch.quint8)
        try:
            # Correct version with named parameters for scale and zero_point
            correct_output = qF.conv2d(
                q_inputs,
                q_filters,
                self.bias,
                padding=self.padding,
                scale=self.scale,
                zero_point=self.zero_point
            )
            
            # Incorrect version from the issue's example (parameter order)
            incorrect_output = qF.conv2d(
                q_inputs,
                q_filters,
                self.bias,
                self.scale,  # Invalid: scale passed as stride (int expected)
                self.zero_point,  # Invalid: zero_point passed as padding (int expected)
                padding=self.padding  # Overwrites but still invalid
            )
            
            # Compare dequantized outputs
            return torch.allclose(
                correct_output.dequantize(),
                incorrect_output.dequantize(),
                atol=1e-5
            )
        except Exception:
            # Return False if any error occurs (e.g., invalid parameters)
            return torch.tensor([False])

def my_model_function():
    return MyModel()

def GetInput():
    inputs = torch.randn(1, 4, 5, 5, dtype=torch.float)
    return torch.quantize_per_tensor(inputs, scale=1.0, zero_point=0, dtype=torch.quint8)

