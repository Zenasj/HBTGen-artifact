# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape (e.g., (32, 32) for 2D tensors)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        if torch.backends.cuda.matmul.allow_tf32:
            return x.sin() + 1
        else:
            return x.cos() - 1

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a 2D float32 tensor compatible with the model's input expectations
    return torch.rand(32, 32, dtype=torch.float32)

