# torch.rand(B, C), torch.randint(0, C, (B,), dtype=torch.uint8)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.NLLLoss()
        
    def forward(self, inputs):
        log_probs, target = inputs
        return self.loss(log_probs, target)

def my_model_function():
    return MyModel()

def GetInput():
    B, C = 2, 3  # Example batch size and number of classes
    log_probs = torch.rand(B, C)
    target = torch.randint(0, C, (B,), dtype=torch.uint8)
    return (log_probs, target)

