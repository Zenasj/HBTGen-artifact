import torch
import torch.nn as nn

def test_return_empty(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return y

        ep = torch.export.export(M(), (torch.randn(3), {}))

class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[3]"):
            return ()