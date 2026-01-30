import torch
import torch.nn as nn

class M(torch.nn.Module):
        def forward(self, x, flag):
            flag = flag.item()

            def true_fn(x):
                return x.clone()

            return torch.cond(flag > 0, true_fn, true_fn, [x])