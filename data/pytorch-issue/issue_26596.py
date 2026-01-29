# torch.rand(11, 5, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, n, d, m):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(n, d, max_norm=True)
        self.W = nn.Parameter(torch.randn((m, d), requires_grad=True))
        self.optimizer = torch.optim.Adam(list(self.embedding.parameters()) + [self.W], lr=1e-3)

    def forward(self, idx):
        # Clone the weight tensor to avoid in-place modification issues
        embedding_weight = self.embedding.weight.clone()
        a = embedding_weight @ self.W.t()  # Line a
        b = self.embedding(idx) @ self.W.t()  # Line b
        out = (a.unsqueeze(0) + b.unsqueeze(1))
        loss = out.sigmoid().prod()
        return loss

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(n=3, d=5, m=7)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch_size = 11
    idx = torch.randint(0, 3, (batch_size,))
    return idx

