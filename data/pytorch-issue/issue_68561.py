import torch

N = 41.0
(N* torch.tensor([1.0/N], dtype=torch.float32)).int()

(41.0 * torch.tensor([1.0/41.0], dtype=torch.float32)).int()
tensor([0], dtype=torch.int32)
(82.0 * torch.tensor([1.0/82.0], dtype=torch.float32)).int()
tensor([0], dtype=torch.int32)

(41.0 * torch.tensor([1.0/41.0], dtype=torch.float64)).int()
tensor([1], dtype=torch.int32)

N = 41.0
(N* torch.tensor([1.0/N], dtype=torch.float32)).item()
0.9999999403953552