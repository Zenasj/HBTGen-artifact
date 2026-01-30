import torch
import torch.nn as nn

def test_export_colin(self):
        class MyModel(torch.nn.Module):
            def forward(self, numel, scalar):
                u0 = numel.item()
                torch._check_is_size(u0)
                x = torch.ones(u0 + 1)
                return scalar - x

        model = MyModel().eval().cuda()
        numel = torch.tensor(10)
        scalar = torch.randn(1,)
        torch.export.export(model, (numel, scalar))