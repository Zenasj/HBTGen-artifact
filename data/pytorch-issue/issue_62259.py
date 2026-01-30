import torch
import sys

print('Python version', sys.version)
print('Torch version', torch.__version__)
i = torch.tensor([[0, 1, 1],
                  [2, 0, 2]])
v = torch.tensor([3, 4, 5], dtype=torch.float32)
T = torch.sparse_coo_tensor(i, v, [2, 4])
T.coalesce()