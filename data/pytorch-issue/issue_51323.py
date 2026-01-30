import torch

# example from docs

z = torch.zeros(2, 4, dtype = torch.int32).scatter_(1, torch.tensor([[2], [3]]), 1)
# works

z = torch.zeros(2, 4, dtype = torch.int32).scatter_(1, torch.tensor([[2], [3]]), 1, reduce="add")
# RuntimeError: scatter_(): Expected floating or complex type for self.