# torch.rand(5, 5, dtype=torch.float)  # inferred input shape (sparse matrix)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense_weight = nn.Parameter(torch.randn(5, 5))  # 5x5 dense matrix
    
    def forward(self, x):
        return torch.sparse.mm(x, self.dense_weight)

def my_model_function():
    return MyModel()

def GetInput():
    indices = torch.tensor([[0, 2, 4], [1, 3, 0]])
    values = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    sparse_input = torch.sparse_coo_tensor(indices, values, (5, 5))
    return sparse_input

