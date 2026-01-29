# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn
import types

class MyModel(nn.Module):
    class FooSub(nn.Module):
        def __init__(self):
            super().__init__()
            self.foo = True  # Attribute for conditional in forward
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Replicates MyModuleFoo's behavior from the issue
            return x * 123 if self.foo else x * 0

    class TrainSub(nn.Module):
        # Replicates MyModuleTraining's behavior using self.training
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 123 if self.training else x * 0

    def __init__(self):
        super().__init__()
        # Encapsulate both test cases as submodules
        self.foo_sub = MyModel.FooSub()
        self.train_sub = MyModel.TrainSub()

    def forward(self, x: torch.Tensor):
        # Execute both submodules and return their outputs
        return self.foo_sub(x), self.train_sub(x)

def my_model_function():
    # Returns an instance with initialized submodules
    return MyModel()

def GetInput():
    # Returns tensor matching the input shape (1,) used in the repro
    return torch.rand(1, dtype=torch.float32)

