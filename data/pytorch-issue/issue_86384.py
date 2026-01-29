# torch.rand(50, 16, 20, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        log_probs = x.log_softmax(2)
        targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
        input_lengths = torch.full((16,), 50, dtype=torch.long)
        target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)
        loss = torch.nn.functional.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths
        )
        return loss

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(50, 16, 20)

