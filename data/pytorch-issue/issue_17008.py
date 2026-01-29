# torch.rand(4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        incorrect = x.clone()
        correct = x.clone()
        
        # Incorrect swap using Pythonic syntax (fails to swap properly)
        incorrect[0], incorrect[1] = incorrect[1], incorrect[0]
        
        # Correct swap using temporary variable
        temp = correct[0].clone()
        correct[0] = correct[1]
        correct[1] = temp
        
        # Return boolean indicating if swaps differ
        return torch.tensor([not torch.allclose(incorrect, correct)], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, dtype=torch.float32)

