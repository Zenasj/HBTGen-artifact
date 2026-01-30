import torch

log_input("x", x)
torch.ops.aten.topk(x, 1)
torch.ops.aten.topk(x, 1, 0)
torch.ops.aten.topk(x, 1, sorted=False)
torch.ops.aten.topk(x, 1, 0, sorted=False)