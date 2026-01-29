# torch.rand(B, 1, dtype=torch.int64)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, bits):
        bits_val = bits[0].item()  # Extract integer value from input tensor
        n = int(2 ** bits_val)     # Apply cast to fix float-to-int conversion
        i = 0
        for _ in range(n):
            i += 1
        return torch.tensor(i, dtype=torch.int64)

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random 1-element integer tensor (e.g., bits value between 1-10)
    return torch.randint(1, 10, (1,), dtype=torch.int64)

