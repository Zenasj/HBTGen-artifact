# torch.rand(3, dtype=torch.float32)  # Input shape is (3,)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compare torch.tensor() vs clone().detach() for in-place operations
        t1 = torch.tensor(x)           # Method A
        t2 = x.clone().detach()        # Method B
        
        # Attempt in-place unsqueeze and track success
        success_a = 0
        success_b = 0
        try:
            t1.unsqueeze_(1)
            success_a = 1
        except:
            pass
        
        try:
            t2.unsqueeze_(1)
            success_b = 1
        except:
            pass
        
        return torch.tensor([success_a, success_b], dtype=torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, dtype=torch.float32)

