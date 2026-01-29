# torch.rand(B, C, H, W, dtype=torch.float32)  # Input shape: (1, 4, 256, 256)
import torch
import torch.nn as nn

class ChunkModel(nn.Module):
    def forward(self, x):
        return x.chunk(4, dim=1)

class SplitModel(nn.Module):
    def forward(self, x):
        # Split into chunks of size 1 along dim 1 to match chunk(4) behavior
        return torch.split(x, 1, dim=1)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.chunk_model = ChunkModel()
        self.split_model = SplitModel()
    
    def forward(self, x):
        chunks = self.chunk_model(x)
        splits = self.split_model(x)
        # Compare all outputs between chunk and split implementations
        all_close = True
        for c, s in zip(chunks, splits):
            if not torch.allclose(c, s, atol=1e-6):
                all_close = False
                break
        return torch.tensor(all_close, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 4, 256, 256)

