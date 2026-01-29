# (indices: torch.randint(0, 256, (3,4), dtype=torch.int64), per_sample_weights: torch.rand(3,4, requires_grad=True, device="cuda:0"))
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_bag = nn.EmbeddingBag(256, 256, mode="sum")
        self.embedding_bag.requires_grad_(False)  # Freeze embedding weights to reproduce the error

    def forward(self, inputs):
        indices, per_sample_weights = inputs
        return self.embedding_bag(
            indices,
            per_sample_weights=per_sample_weights
        )

def my_model_function():
    return MyModel()

def GetInput():
    device = "cuda:0"
    indices = torch.arange(1, 5, device=device).expand(3, -1).to(torch.int64)
    per_sample_weights = torch.rand(3, 4, device=device, requires_grad=True)
    return (indices, per_sample_weights)

