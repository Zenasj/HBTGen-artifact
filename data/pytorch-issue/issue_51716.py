import torch
import torch.nn as nn
n = 16
d = 16
a = torch.rand((n, d))
p = nn.functional.softmax(a, dim=-1)
nn.functional.kl_div(p, p)

nn.functional.kl_div(p, p, log_target=True)