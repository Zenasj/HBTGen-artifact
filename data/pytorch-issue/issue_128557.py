import torch

with torch.autograd.detect_anomaly():
  mt = torch.masked.MaskedTensor(
      torch.rand(2, 2),
      torch.rand(2, 2) > 0.5,
      requires_grad=True
  )
  mt.sum().backward()