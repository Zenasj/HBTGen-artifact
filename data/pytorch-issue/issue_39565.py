import torch

x = torch.arange(1,5, device='cpu', dtype=torch.int64).reshape(2, 2)
torch.zeros(2, 2, device=device).scatter_(0, torch.tensor([[1, 0 ], [0, 0, ]], device=device), x)

tensor([[2.8026e-45, 4.2039e-45],
        [1.4013e-45, 0.0000e+00]])

x = torch.arange(1,5, device='cuda:0', dtype=torch.int64).reshape(2, 2)
torch.zeros(2, 2, device=device).scatter_(0, torch.tensor([[1, 0 ], [0, 0, ]], device=device), x)

x = torch.arange(1,5, dtype=torch.float).reshape(2, 2)
torch.zeros(2, 2, device=device).scatter_(0, torch.tensor([[1, 0 ], [0, 0, ]], device=device), x)