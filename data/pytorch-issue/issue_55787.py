# torch.rand(B, C, H, W, dtype=torch.float32)  # Example input shape (B=1, C=3, H=224, W=224)
import torch
from torch import nn
from typing import Optional, Tuple

class MyModel(nn.Module):
    def forward(self, x):
        # Problematic code (causes error when scripted due to multi-line Future annotation)
        future_err: Optional[
            torch.jit.Future[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = None
        # Fixed code using workaround (imported Future explicitly)
        from torch.jit import Future
        future_ok: Optional[Future[Tuple[torch.Tensor]]] = None
        return x  # Dummy return to satisfy nn.Module

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

