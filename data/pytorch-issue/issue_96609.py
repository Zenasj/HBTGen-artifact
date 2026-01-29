import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.rand(1, 1, 2, dtype=torch.float32, device='cuda')  # Input shape matches 'a' tensor's dimensions
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1, 1, 2, device='cuda'))  # Matches first example's 'a'
        self.b = nn.Parameter(torch.rand(1, 1, 2, 3, 1, device='cuda'))  # Matches first example's 'b'
        
    def forward(self, x):
        # Replicate first example's operations (error-prone path)
        c = F.pad(self.a, (0, 1, 0, 0), 'reflect')  # Pad to (1,1,3)
        d = torch.add(self.b, c)  # Broadcast addition
        return F.pad(d, (-2, 0, 0, 0, 0, 0, 0, 0, 0, 1))  # Apply problematic negative padding

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a dummy input matching the expected shape of 'a' (though unused in forward)
    return torch.rand(1, 1, 2, dtype=torch.float32, device='cuda')

