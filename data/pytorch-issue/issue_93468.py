# torch.rand(10, dtype=torch.int64)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    @staticmethod
    def sequence_mask(lengths, max_len=None):
        batch_size = lengths.numel()
        max_len = max_len or lengths.max()
        return ~(torch.arange(0, max_len, device=lengths.device)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))
    
    def forward(self, a):
        return self.sequence_mask(a)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(10, 30, (10,), dtype=torch.int64)

