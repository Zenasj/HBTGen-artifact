# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Valid approach using keepdim (JIT-compatible)
        res_keepdim = x.sum(dim=0, keepdim=True)
        
        # Emulate keepdims behavior via manual unsqueeze (avoids using deprecated keyword)
        res_keepdims_emulated = x.sum(dim=0).unsqueeze(0)
        
        # Return comparison result as a tensor (JIT-compatible)
        return torch.tensor(
            torch.allclose(res_keepdim, res_keepdims_emulated),
            dtype=torch.bool
        )

def my_model_function():
    return MyModel()

def GetInput():
    # 4D tensor matching the input comment structure
    return torch.rand(10, 10, 1, 1, dtype=torch.float32)

