# torch.rand(10, 10, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.register_buffer('scale', torch.randn(1, 10))  # Non-trainable buffer for scale

    def forward(self, x):
        return F.relu(self.linear1(x)) * self.scale

class SuperViaSuper(BasicModule):
    def forward(self, x):
        x = super().forward(x)
        return x + 10.0

class SuperViaDirect(BasicModule):
    def forward(self, x):
        x = BasicModule.forward(self, x)  # Direct parent method call (problematic in Dynamo)
        return x + 10.0

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.via_super = SuperViaSuper()
        self.via_direct = SuperViaDirect()

    def forward(self, x):
        # Reshape to 2D if input is 4D (B, C, H, W)
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        out1 = self.via_super(x)
        out2 = self.via_direct(x)
        # Return boolean tensor indicating if outputs match
        return torch.tensor([torch.allclose(out1, out2)], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, 10, 1, 1)  # Matches the input shape comment

