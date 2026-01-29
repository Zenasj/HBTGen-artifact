# torch.rand(0, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        try:
            amax_val = torch.amax(x)
            max_val = torch.max(x)
            return torch.allclose(amax_val, max_val).to(torch.int32)
        except RuntimeError:
            # Indicates inconsistency when max raises error (empty input)
            return torch.tensor(0, dtype=torch.int32)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(0, 3, dtype=torch.float32)

