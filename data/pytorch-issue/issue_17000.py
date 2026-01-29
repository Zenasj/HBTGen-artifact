# torch.rand(4, 4, dtype=torch.float32)

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.numpy_like = NumpyContains()
        self.exact_match = ExactMatchContains()

    def forward(self, input_tensor):
        query = input_tensor[0]
        main = input_tensor[1:]
        res1 = self.numpy_like(main, query)
        res2 = self.exact_match(main, query)
        return res1 != res2

class NumpyContains(nn.Module):
    def forward(self, main, query):
        flat_main = main.view(-1)
        flat_query = query.view(-1)
        comparison = (flat_main.unsqueeze(0) == flat_query.unsqueeze(1))
        return comparison.any().any()

class ExactMatchContains(nn.Module):
    def forward(self, main, query):
        query_expanded = query.unsqueeze(0)
        matches = torch.all(main == query_expanded, dim=1)
        return matches.any()

def my_model_function():
    return MyModel()

def GetInput():
    main = torch.rand(3, 4)
    query = main[0].clone()
    query[0] += 1  # Ensure exact_match returns False
    input_tensor = torch.cat([query.unsqueeze(0), main], dim=0)
    return input_tensor

