# torch.randint(0, 8, (20,), dtype=torch.long), torch.sort(torch.randint(0, 20, (3,), dtype=torch.long))[0], torch.randn(2, requires_grad=True)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 2), requires_grad=True)  # Matches weight shape in original example

    def embedding_forward(self, input_indices, offsets):
        return F.embedding_bag(
            input_indices,
            self.weight,
            offsets=offsets,
            mode='mean',
            sparse=False,
            scale_grad_by_freq=False,
            max_norm=-1.0  # Key parameter causing the issue
        )
    
    def f_forward(self, x):
        # Reproduces the second minimal example's in-place mutation pattern
        temp = x.detach()
        temp.add_(1)
        return x * 2

    def forward(self, inputs):
        input_indices, offsets, x = inputs  # Unpack both scenarios' inputs
        emb_out = self.embedding_forward(input_indices, offsets)
        f_out = self.f_forward(x)
        return emb_out, f_out  # Return both outputs for error observation

def my_model_function():
    return MyModel()

def GetInput():
    # Generate inputs for both scenarios:
    # 1. EmbeddingBag scenario inputs
    input_indices = torch.randint(0, 8, (20,), dtype=torch.long)  # Matches indices in original example
    offsets = torch.sort(torch.randint(0, 20, (3,), dtype=torch.long))[0]  # Ensure sorted offsets
    
    # 2. Minimal in-place mutation scenario input
    x = torch.randn(2, requires_grad=True)  # Matches second example's input
    
    return (input_indices, offsets, x)

