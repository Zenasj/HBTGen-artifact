# torch.rand(B, 2, dtype=torch.float32, device='cuda')
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)  # Matches input tensor size from issue's example

    def forward(self, x):
        return torch.relu(self.fc(x))  # Simple forward pass using CUDA-capable ops

def my_model_function():
    model = MyModel()
    model.to('cuda')  # Ensure model is on CUDA device
    return model

def GetInput():
    # Generate input matching the model's expected dimensions and device
    return torch.rand(1, 2, dtype=torch.float32, device='cuda')

