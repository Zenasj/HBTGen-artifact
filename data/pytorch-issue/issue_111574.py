# torch.rand(4, 16, dtype=torch.float32, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Construct a minimal graph with non-contiguous sparse CSR values to trigger the error
        num_nodes = 4
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long, device='cuda')
        edge_index_sorted = edge_index[:, torch.argsort(edge_index[0])]
        row, col = edge_index_sorted
        crow_indices = torch.bincount(row, minlength=num_nodes + 1).cumsum(0).to('cuda')
        
        # Create non-contiguous values to simulate the PyG bug scenario
        values = torch.rand(6, device='cuda')[1:5]  # Sliced tensor to ensure non-contiguous storage
        
        self.adj = torch.sparse_csr_tensor(
            crow_indices=crow_indices,
            col_indices=col,
            values=values,
            size=(num_nodes, num_nodes),
            device='cuda'
        )
        self.deg = crow_indices[1:] - crow_indices[:-1]

    def forward(self, x):
        # Reproduce the error-prone operation from PyG's spmm
        out = torch.sparse.mm(self.adj, x)
        out = out / self.deg.view(-1, 1).clamp_(min=1)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 16, dtype=torch.float32, device='cuda')

