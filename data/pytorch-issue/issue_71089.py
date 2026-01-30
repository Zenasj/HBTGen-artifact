import torch

# segfaults in torch 1.10, works as expected in torch 1.9:
tensor_nan = torch.tensor([float("nan")])
assert torch.unique(tensor_nan, dim=0).numel() == 1
assert torch.isnan(torch.unique(tensor_nan, dim=0)[0])

# works as expected in both torch 1.10 and 1.9:
tensor_nan = torch.tensor([float("nan")])
assert torch.unique(tensor_nan, dim=None).numel() == 1
assert torch.isnan(torch.unique(tensor_nan, dim=None)[0])

# works as expected in both torch 1.10 and 1.9:
tensor_inf = torch.tensor([float("inf")])
assert torch.unique(tensor_inf, dim=0).numel() == 1
assert torch.isinf(torch.unique(tensor_inf, dim=0)[0])