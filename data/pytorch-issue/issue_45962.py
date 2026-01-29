# torch.rand(10, dtype=torch.float32)
import torch
from torch import nn
import tempfile
import os

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create temporary file and write data to replicate the bug scenario
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.temp_filename = f.name
            data = torch.zeros(10, dtype=torch.uint8)  # Example data
            f.write(data.numpy().tobytes())
        
        # Load tensor using from_file with shared=False (problematic scenario)
        self.loaded_tensor = torch.from_file(
            self.temp_filename,
            shared=False,
            size=data.numel(),
            dtype=torch.uint8
        )
        
        # Attempt to delete file - should fail due to open file mapping
        try:
            os.remove(self.temp_filename)
        except PermissionError:
            # File remains open due to THMapAllocator (Windows-specific issue)
            pass

    def forward(self, x):
        # Example computation using the loaded tensor (sum for demonstration)
        return x + self.loaded_tensor.sum().float()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, dtype=torch.float32)

