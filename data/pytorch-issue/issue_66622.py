import torch.nn as nn

import torch

try:
    t = torch.randn(2, 1, dtype=torch.float64, requires_grad=True)
    idx = torch.tensor([0, 1])
    torch.autograd.gradcheck(lambda idx, t : torch.nn.functional.embedding(idx, t, padding_idx=1), (idx, t, ))
except Exception as e:
    print("PADDING IDX:", e)

try:
    t = torch.ones(2, 1, dtype=torch.float64, requires_grad=True)
    idx = torch.tensor([0, 1])
    torch.autograd.gradcheck(lambda idx, t : torch.nn.functional.embedding(idx, t, max_norm=1.), (idx, t, ))
except Exception as e:
    print("MAX NORM:", e)

try:
    t = torch.randn(2, 1, dtype=torch.float64, requires_grad=True)
    idx = torch.tensor([0, 1, 1])
    torch.autograd.gradcheck(lambda idx, t : torch.nn.functional.embedding(idx, t, scale_grad_by_freq=True), (idx, t, ))
except Exception as e:
    print("SCALE GRAD BY FREQUENCY:", e)

try:
    t = torch.randn(2, 1, dtype=torch.float64, requires_grad=True)
    idx = torch.tensor([0, 1])
    torch.autograd.gradcheck(lambda idx, t : torch.nn.functional.embedding(idx, t, sparse=True), (idx, t, ))
except Exception as e:
    print("SPARSE", e)