import torch

T = torch.empty(3, 0, 4)
result_mean = torch.mean(T, dim=1, keepdim=True)
result_sum = torch.sum(T, dim=1, keepdim=True)

print(result_mean.shape)  # Expected: torch.Size([3, 0, 4]), Actual: torch.Size([3, 1, 4]) and filled with NaN
print(result_sum.shape)   # Expected: torch.Size([3, 0, 4]), Actual: torch.Size([3, 1, 4]) and filled with 0