import torch
import torch.nn as nn

def test_slicing(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                b = x.item()
                torch._check_is_size(b)
                torch._check(b < y.shape[0])
                return y[0, b]
        
        print(torch.export.export(M(), (torch.tensor(4), torch.ones(10, 10))))