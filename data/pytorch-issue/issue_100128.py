import torch
import torch.nn as nn

class F(torch.nn.Module):
            def forward(self, x, y):
                x = x.t_()
                y = y.t_()
                return (x + y,)