# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class Net1(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a  # Mimics original forward behavior with two inputs

class Net2(nn.Module):
    def forward(self, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return c  # Mimics original forward behavior with two inputs

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = Net1()  # Simulated "remote" module on rank0
        self.net2 = Net2()  # Simulated "remote" module on rank1

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Replicates the workaround behavior from the issue:
        # Passing (input, input) to mimic RPC argument handling
        outputs1 = self.net1(input, input)
        outputs2 = self.net2(outputs1, outputs1)
        return outputs2

def my_model_function():
    return MyModel()

def GetInput():
    # Random tensor matching expected input (B, C, H, W)
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

