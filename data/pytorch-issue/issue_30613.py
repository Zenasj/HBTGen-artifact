# torch.rand(1, dtype=torch.float32)  # dummy input, not used in forward
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # First scenario: float division (should produce inf)
        a = torch.tensor([1.0], dtype=torch.float)
        b = torch.tensor([0.0], dtype=torch.float)
        res1 = torch.div(a, b)
        
        # Second scenario: integer division (should raise error)
        c = torch.tensor([1], dtype=torch.int64)
        d = torch.tensor([0], dtype=torch.int64)
        try:
            res2 = torch.div(c, d)
            # Compare outputs (inf vs error implies inconsistency)
            same = torch.tensor(False, dtype=torch.bool)
        except:
            # If second division errors, results are inconsistent
            same = torch.tensor(False, dtype=torch.bool)
        return same

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1)  # Dummy input (not used in forward)

