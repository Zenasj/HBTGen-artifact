# torch.randint(0, 100, (57,), dtype=torch.int64)
import torch
from torch import nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Weight tensor (vocabulary size 100, embedding dim 5)
        self.weight = nn.Parameter(torch.rand(100, 5, dtype=torch.float32), requires_grad=False)
        # Predefined offsets with last value < input length (53 < 57)
        self.register_buffer('offsets', torch.tensor(
            [0, 6, 12, 15, 25, 32, 40, 42, 46, 53, 53], 
            dtype=torch.int64
        ))
    
    def forward(self, input):
        return F.embedding_bag(
            input,
            self.weight,
            self.offsets,
            norm_type=2.0,
            scale_grad_by_freq=False,
            mode='mean',
            sparse=True,
            include_last_offset=True,
            padding_idx=61
        )

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input tensor matching the problematic length (57 elements)
    return torch.randint(0, 100, (57,), dtype=torch.int64)

