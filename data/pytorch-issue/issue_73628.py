# torch.rand(1, 210120, dtype=torch.float32)
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k = 9
        self.dim = 0

    def forward(self, x):
        # Compare old (segfault) vs new (error) behavior via exception handling
        try:
            result = torch.kthvalue(x, k=self.k, dim=self.dim)
            # Old behavior would segfault, but returning 0 indicates no error (bad case)
            return torch.tensor(0)
        except RuntimeError:
            # New behavior throws error, return 1 to indicate proper handling (good case)
            return torch.tensor(1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 210120, dtype=torch.float32)

