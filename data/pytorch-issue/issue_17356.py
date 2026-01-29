# torch.rand(B, 2, dtype=torch.float32, device='cuda')  # Inferred input shape from the issue's example
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(10, 3, sparse=True)
        self.net = nn.Linear(2, 3)

    def forward(self, x):
        # Fixed index 0 for embedding as per original model's forward logic
        return self.net(x) + self.embedding(torch.tensor(0, device=x.device))

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a CUDA tensor matching the input shape expected by MyModel
    return torch.rand(20, 2, dtype=torch.float32, device='cuda')

