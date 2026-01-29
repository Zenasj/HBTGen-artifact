# torch.rand(2, 5, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_w = nn.Parameter(torch.Tensor(3, 5))
        nn.init.normal_(self.enc_w)

    def encode(self, data):
        return F.linear(data, self.enc_w)

    def decode(self, data):
        # Use self.enc_w.T directly to avoid the issue of not moving self.dec_w to the correct device
        return F.linear(data, self.enc_w.T)

    def forward(self, data):
        return self.decode(self.encode(data))

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand((2, 5), dtype=torch.float32)

