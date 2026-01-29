# torch.rand(1, dtype=torch.float32)
import torch
import torch.nn as nn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(0.0))  # Single parameter being optimized
        self.optimizer = SGD([self.param], lr=0.1)
        self.scheduler = ReduceLROnPlateau(self.optimizer, verbose=True, patience=3)
    
    def forward(self, loss):
        """Passes loss through while maintaining model state"""
        return loss

def my_model_function():
    return MyModel()

def GetInput():
    """Returns a scalar tensor compatible with the model's expected input"""
    return torch.rand(1, dtype=torch.float32)

