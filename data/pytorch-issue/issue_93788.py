# torch.rand(2, dtype=torch.int64, device='cuda')  # Inferred input shape based on the scaling calculation example
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Scaling factor calculation (source of precision discrepancy)
        min_val = x.min()
        max_val = x.max()
        scale1 = 800.0 / min_val
        scale2 = 1333.0 / max_val
        scale = torch.min(scale1, scale2)
        
        # Cat operation with mixed dtypes (source of dtype assertion error)
        a = torch.tensor([1], dtype=torch.int32, device=x.device)
        b = torch.tensor([2.0], dtype=torch.float32, device=x.device)
        cat_result = torch.cat([a, b], dim=0)  # This will trigger a dtype mismatch error
        
        return scale, cat_result

def my_model_function():
    return MyModel()

def GetInput():
    # Input that triggers both scaling and cat operations
    return torch.tensor([459, 640], dtype=torch.int64, device='cuda')

