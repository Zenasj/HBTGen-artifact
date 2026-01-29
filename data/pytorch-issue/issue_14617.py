# torch.rand(4, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, input_dense):
        # Convert input to sparse tensor
        idx = input_dense.nonzero()
        vals = input_dense[input_dense != 0]
        sparse_adj = torch.sparse.FloatTensor(idx.t(), vals, input_dense.size())
        sparse_adj = sparse_adj.coalesce()
        
        # Compute diagonal matrix D
        ones = torch.ones(input_dense.size(0), 1, device=input_dense.device)
        D_vec = sparse_adj.mm(ones).view(-1)
        D_vec = D_vec ** -0.5
        D = torch.diag(D_vec)
        
        # Compute sparse and dense results
        sparse_result = sparse_adj.t().mm(D).t()
        dense_adj = sparse_adj.to_dense()
        dense_result = dense_adj.t().mm(D).t()
        
        # Return 1.0 if results match, else 0.0 (for compatibility with torch.compile)
        return torch.tensor([1.0], dtype=torch.float32) if torch.allclose(sparse_result, dense_result) else torch.tensor([0.0], dtype=torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 4, dtype=torch.float32)

