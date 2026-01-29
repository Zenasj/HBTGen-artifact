# torch.rand(64, 768, 768, dtype=torch.float16)  # Inferred input shape

import torch
from torch import nn
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

device = "cuda"
SparseSemiStructuredTensor._FORCE_CUTLASS = True

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(768, 3072),
            nn.Linear(3072, 768),
        ).half().to(device)

        for i in range(2):
            m, n = self.model[i].weight.shape
            mask = torch.Tensor([0, 0, 1, 1]).tile(m, n // 4).to(device).bool()
            self.model[i].weight = nn.Parameter(self.model[i].weight * mask)

        for i in range(2):
            self.model[i].weight = nn.Parameter(to_sparse_semi_structured(self.model[i].weight))

    def forward(self, x):
        return self.model(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(64, 768, 768, dtype=torch.float16, device=device)

