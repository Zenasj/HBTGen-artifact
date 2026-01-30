import torch

LD = torch.tensor([[1.0, 2.0, 3.0],
                    [2.0, 5.0, 6.0],
                    [3.0, 6.0, 9.0]], dtype=torch.float32)
pivots = torch.tensor([0, 1, 2], dtype=torch.int32)
B = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
torch.linalg.ldl_solve(LD, pivots, B, hermitian=True)