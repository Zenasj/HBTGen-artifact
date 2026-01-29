# torch.rand(3, dtype=torch.float32).cuda()  # Inferred input shape is (3,)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.weight = nn.Parameter(torch.rand(3, requires_grad=True))  # Matches original code's weight
        self.register_buffer('target', torch.randint(0, 3, (3,), dtype=torch.long))  # Fixed target tensor

    def forward(self, input):
        # Compute NLL loss with fixed target
        loss1 = F.nll_loss(input, self.target)
        # Multiply by weight (shape 3), sum, and scale by 0.25 as in original code
        loss = torch.sum(self.weight * loss1) * 0.25
        return loss

def my_model_function():
    model = MyModel()
    model.cuda()  # Move to CUDA as in the issue's example
    return model

def GetInput():
    # Return a random tensor matching the input shape (3,) with requires_grad=True on CUDA
    return torch.randn(3, requires_grad=True).cuda()

