import torch

a = torch.eye(3).to(device='cuda:0')
try:
    torch.linalg.cholesky(a)   # Throws error, see error message in bottom
except Exception as e:
    print(e)

torch.linalg.cholesky(a)   # When run for the second time, doesn't throw error