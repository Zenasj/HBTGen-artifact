# torch.rand(2, dtype=torch.float32)  # Inferred from the provided example in the issue's comments
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Reproduces the core operation causing the quantize_per_tensor error in Dynamo
        qx = torch.quantize_per_tensor(x, scale=0.1, zero_point=5, dtype=torch.quint8)
        return qx.dequantize()  # Ensure output is a float Tensor for compatibility

def my_model_function():
    # Returns an instance of the minimal model triggering the Dynamo error
    return MyModel()

def GetInput():
    # Returns a 1D tensor matching the example in the issue's comments
    return torch.rand(2, dtype=torch.float32)

