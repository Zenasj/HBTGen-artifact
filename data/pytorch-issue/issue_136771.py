import torch
import torch.nn as nn

# torch.rand(B, C, H, W, dtype=torch.float32)
class MyModel(nn.Module):
    def forward(self, x):
        if torch.is_grad_enabled():
            # Simulate autograd-enabled custom op (e.g., with gradient computation)
            # Placeholder: Returns x + 1.0 (replace with actual custom op call)
            return x + 1.0  # Actual implementation would call torch.ops.myop_autograd(x)
        else:
            # Simulate non-autograd custom op (no gradient computation)
            # Placeholder: Returns x as-is (replace with actual custom op call)
            return x  # Actual implementation would call torch.ops.myop_noautograd(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random 4D tensor (B=1, C=3, H=224, W=224) matching typical input expectations
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

