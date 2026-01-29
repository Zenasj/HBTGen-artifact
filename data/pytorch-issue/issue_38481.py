# torch.rand(3, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, d, idx, device):
        super().__init__()
        # Assumptions: idx is a list of [row, col] pairs for sparse connections
        # Example: idx = [[0,1], [1,2], [2,0]] for a 3x3 matrix (simplified for testing)
        i = torch.LongTensor(idx).t()
        size = (3, 3)  # Reduced size for demonstration; original was 4,847,571x4,847,571
        values_M = torch.ones(len(idx), dtype=torch.float32)
        M = torch.sparse.FloatTensor(i, values_M, torch.Size(size))
        values_temp = torch.full((len(idx),), (1 - d)/size[0], dtype=torch.float32)
        temp = torch.sparse.FloatTensor(i, values_temp, torch.Size(size))
        M_hat = d * M + temp
        self.M_hat = M_hat.to(device)
    
    def forward(self, v):
        return torch.mm(self.M_hat, v)

def my_model_function():
    # Assumptions: damping factor d=0.85 (common PageRank value)
    # Simplified indices for a small 3x3 matrix (original used LiveJournal graph)
    d = 0.85
    idx = [[0, 1], [1, 2], [2, 0]]  # Example connections for 3-node graph
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return MyModel(d, idx, device)

def GetInput():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Matches the 3x3 matrix size from the model's sparse tensors
    return torch.rand(3, 1, dtype=torch.float32, device=device)

