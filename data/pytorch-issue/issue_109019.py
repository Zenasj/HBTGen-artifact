import torch
from torch import nn

# torch.rand(1, 4, 3, dtype=torch.float32)  # Inferred input shape from the example
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('seq_len', torch.tensor([-1], dtype=torch.int64))

    def forward(self, logits):
        full_default = torch.full((1,), 0, dtype=torch.int64, device=logits.device)
        return logits[full_default, self.seq_len]

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 4, 3, dtype=torch.float32)

