# torch.rand(B, S, dtype=torch.long)
import torch
import torch.nn as nn

class Model0(nn.Module):
    def forward(self, input_ids):
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long).unsqueeze(0)
        position_ids = position_ids.expand(input_ids.size())
        return position_ids

class Model1(nn.Module):
    def forward(self, input_ids):
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long).unsqueeze(0)
        position_ids = position_ids.repeat(input_ids.size(0), 1)
        return position_ids

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model0 = Model0()
        self.model1 = Model1()
    
    def forward(self, input_ids):
        out0 = self.model0(input_ids)
        out1 = self.model1(input_ids)
        # Return boolean tensor indicating if outputs are identical
        return torch.all(out0 == out1)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches benchmark's sample input dimensions (B=32, S=512)
    return torch.randint(0, 10, (32, 512), dtype=torch.long)

