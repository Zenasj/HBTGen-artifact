# torch.rand(4, 2, dtype=torch.float).cuda() ← Add a comment line at the top with the inferred input shape
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.S = nn.Parameter(
            torch.sparse_coo_tensor(
                indices=torch.tensor([[0, 0, 1, 2], [2, 3, 0, 3]]),
                values=torch.tensor([1.0, 2.0, 1.0, 3.0]),
                size=[3, 4]
            ).to_dense(), requires_grad=False
        )
        self.fc = nn.Linear(6, 4)

    def forward(self, x):
        x = torch.mm(self.S, x)
        x = x.reshape(-1)
        x = self.fc(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(4, 2, dtype=torch.float).cuda()

