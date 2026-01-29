import torch
import torch.nn as nn

# torch.rand(B, C, dtype=torch.float16, device="cuda")
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 20)  # From original issue's example
        self.eps = 1e-12
        self.dim = -1
        
    def forward(self, x):
        denom = x.norm(2.0, self.dim, keepdim=True)
        print("denom", denom.dtype)
        return denom

def my_model_function():
    model = MyModel()
    model.eval()
    model.to(torch.float16).to("cuda")  # Matches issue's model setup
    return model

def GetInput():
    return torch.rand(8, 10, dtype=torch.float16, device="cuda")  # Matches example input shape and dtype

