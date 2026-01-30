import torch
import torch.nn as nn

def test_mm_cat(self):

        @torch.jit.script
        def func(x, y, w, b):
            l = [x, y]
            z = torch.cat(l, dim = 0)
            li = torch.torch.nn.functional.linear(z, w, b)
            return torch.nn.functional.layer_norm(li, li.size())

        x = torch.rand(5, 20)
        y = torch.rand(5, 20)
        w = torch.rand(40, 20)
        b = torch.rand(40)

        with num_profiled_runs(1):
            func(x, y, w, b)
            func(x, y, w, b)
            func(x, y, w, b)