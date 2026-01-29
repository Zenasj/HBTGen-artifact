# torch.rand(B, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Reproduce the sparse COO tensor setup from the original issue
        indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
        values = torch.tensor([1.0, 2.0], dtype=torch.float32)
        self.a = nn.Parameter(
            torch.sparse_coo_tensor(indices, values, size=(2, 2), dtype=torch.float32),
            requires_grad=False
        )

    def forward(self, x):
        # Minimal forward pass using the sparse parameter (for torch.compile compatibility)
        # Expand input to 2D for matrix multiplication
        x = x.unsqueeze(-1)  # (batch, 2) → (batch, 2, 1)
        result = torch.sparse.mm(self.a, x)
        return result.squeeze(-1)  # Return (batch, 2) tensor

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random tensor matching the expected input shape
    return torch.rand(2, dtype=torch.float32)

