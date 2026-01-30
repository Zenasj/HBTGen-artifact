import torch

# This works
torch.linalg.pinv(torch.eye(5, device='cpu'))

# RuntimeError: cusolver error: 7, when calling `cusolverDnCreate(handle)`
torch.linalg.pinv(torch.eye(5, device='cuda'))