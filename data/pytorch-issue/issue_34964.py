# torch.rand(1000, 1, dtype=torch.float)  # Input shape is (input_length, 1)

import torch
import torch.nn as nn

def make_sparse_weights(input_length):
    values = torch.ones(input_length, dtype=torch.float)
    indices = torch.tensor([[0, i] for i in range(input_length)], dtype=torch.long).t()
    return torch.sparse.FloatTensor(indices, values, size=(1, input_length))

class SparseNetwork(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.weights = nn.Parameter(make_sparse_weights(input_length), requires_grad=True)

    def forward(self, x):
        return torch.norm(torch.sparse.mm(self.weights, x)) ** 2

class DenseNetwork(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.weights = nn.Parameter(make_sparse_weights(input_length).to_dense(), requires_grad=True)

    def forward(self, x):
        return torch.norm(torch.mm(self.weights, x)) ** 2

class MyModel(nn.Module):
    def __init__(self, input_length=1000):
        super().__init__()
        self.sparse_net = SparseNetwork(input_length)
        self.dense_net = DenseNetwork(input_length)

    def forward(self, x):
        return self.sparse_net(x), self.dense_net(x)

def my_model_function():
    return MyModel()

def GetInput():
    input_length = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.rand(input_length, 1, dtype=torch.float, device=device)

