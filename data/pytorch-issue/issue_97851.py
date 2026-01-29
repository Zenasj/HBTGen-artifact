# torch.rand(B, C, H, W, dtype=...)  # Not applicable for this model

import torch
from torch import nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.logits = torch.tensor([0.1, 0.2, 0.3, 0.4]).log()  # Log of the probabilities

    def forward(self, x):
        return F.gumbel_softmax(x, tau=0.1, hard=True)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.tensor([0.1, 0.2, 0.3, 0.4]).log()  # Log of the probabilities

