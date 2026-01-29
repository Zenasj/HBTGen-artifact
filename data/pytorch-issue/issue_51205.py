# torch.rand(1, n, dtype=torch.float32)  # Inferred input shape is (1, n) where n ∈ [16, 100]
import torch
import numpy as np
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Dummy model for demonstration purposes
        self.identity = nn.Identity()
    
    def forward(self, x):
        return self.identity(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Apply workaround for Windows numpy int32 overflow in seed generation
    state = np.random.RandomState(2809)
    # Use dtype=np.int64 to avoid overflow
    seed_val = state.randint(0, 2**63, dtype=np.int64)
    torch.manual_seed(seed_val)
    
    # Generate tensor size 'n' between 16 and 100 (as per Fuzzer parameters)
    n = torch.randint(16, 101, (1,)).item()
    
    # Create input tensor matching Fuzzer's specifications
    return torch.rand(1, n, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

