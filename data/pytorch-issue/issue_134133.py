import torch
import torch.nn as nn

def test_spinning_floordiv():
    class M(torch.nn.Module):
        def forward(self, x):
            total = sum(t.item() for t in x)
            print(total)
            return total // 5

    m = M()
    inp = [torch.tensor(i + 3) for i in range(100)]
    torch.export.export(m, (inp,), strict=False)