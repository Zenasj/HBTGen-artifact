import torch

z = EWDNet(tensor).squeeze(0)
conv = z@z.transpose(-1, -2)
L = torch.linalg.cholesky(conv)

z = torch.randn(1, 32, 100, device=device)
conv = z@z.transpose(-1, -2)
L = torch.linalg.cholesky(conv)