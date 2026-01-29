# torch.rand(1, 10, dtype=torch.float32)  # Inferred input shape based on model's linear layer
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_a = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )
        self.model_b = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )
        # Initialize weights to create discrepancy between the two models
        torch.manual_seed(0)
        self.model_a[0].weight.data.normal_()
        self.model_b[0].weight.data.normal_(mean=1.0)  # Different initialization

    def forward(self, x):
        out_a = self.model_a(x)
        out_b = self.model_b(x)
        # Check if forward outputs differ (core comparison logic from the issue)
        forward_diff = not torch.allclose(out_a, out_b, atol=1e-5, rtol=1e-5)
        return torch.tensor([forward_diff], dtype=torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 10, dtype=torch.float32)

