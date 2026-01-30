import torch.nn as nn

import torch
x1 = torch.rand([1, 1, 1], dtype=torch.float32)
x2 = torch.rand([0, 12, 5676819769219801604], dtype=torch.float32)
res = torch._euclidean_dist(
    x1=x1,
    x2=x2,
)

# multilabel_margin_loss
import torch
input = torch.rand([3, 2], dtype=torch.float32)
target = torch.rand([9, 9, 2, 15, 0, 904932433365235466], dtype=torch.float32)
reduction = 53
res = torch._C._nn.multilabel_margin_loss(
    input=input,
    target=target,
    reduction=reduction,
)

# linalg_pinv
import torch
input = torch.rand([0, 2786357852644344240, 11, 15, 10, 16], dtype=torch.float32)
res = torch._C._linalg.linalg_pinv(
    input=input,
)

# linalg_matmul
import torch
input = torch.rand([2, 10], dtype=torch.float32)
other = torch.rand([8, 0, 9, 13, 1356914755596406577], dtype=torch.float32)
res = torch._C._linalg.linalg_matmul(
    input=input,
    other=other,
)