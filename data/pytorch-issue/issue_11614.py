# torch.randint(0, 3, (5, 4, 1), dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, num_classes=3, dtype=torch.float):
        super().__init__()
        self.num_classes = num_classes
        self.dtype = dtype

    def forward(self, indices):
        eye = torch.eye(self.num_classes, dtype=self.dtype, device=indices.device)
        # Squeeze the last dimension (since input has shape (..., 1))
        flat_indices = indices.squeeze(-1)
        return eye[flat_indices]

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 3, (5, 4, 1), dtype=torch.long)

