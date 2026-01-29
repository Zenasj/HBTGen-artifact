# torch.rand(2, 2, 3, dtype=torch.double)  # Input shape: (T=2, N=2, C=3)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('labels', torch.tensor([1, 2], dtype=torch.int32))
        self.register_buffer('label_sizes', torch.tensor([2, 0], dtype=torch.int32))  # target lengths
        self.register_buffer('input_sizes', torch.tensor([2, 2], dtype=torch.int32))  # input lengths
        self.loss_fn = nn.CTCLoss(zero_infinity=True, reduction='sum')

    def forward(self, inputs):
        log_probs = inputs.log_softmax(2)  # Apply along the last dimension
        return self.loss_fn(log_probs, self.labels, self.input_sizes, self.label_sizes)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 2, 3, dtype=torch.double)

