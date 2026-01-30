import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class Model(nn.Module):
    def __init__(self, n: int, use_cp: bool, allow_first: bool, allow_last: bool):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n = n
        self.use_cp = use_cp
        self.allow_first = allow_first
        self.allow_last = allow_last
        for i in range(self.n):
            self.layers.append(nn.Linear(256, 256))

    def forward(self, x):
        for i in range(self.n):
            if (
                not self.use_cp
                or (i == 0 and not self.allow_first)
                or (i == self.n - 1 and not self.allow_last)
            ):
                print("No checkpoint", i)
                x = self.layers[i](x)
            else:
                print("Checkpointing", i)
                x = checkpoint(self.layers[i], x)
        return x


def test(use_cp, first, last):
    model = Model(4, use_cp, first, last).cuda()
    x = torch.randn(17, 256).cuda()
    loss = model(x).sum()
    try:
        loss.backward()
    except RuntimeError:
        return "RuntimeError"
    return sum([p.grad is None for p in model.parameters()])


print("None grads with NO grad checkpoint:", test(False, False, False))
print()
print("None grads with ALL grad checkpoint (1..n):", test(True, True, True))
print()
print("None grads with grad checkpoint (no first; 2..n):", test(True, False, True))
print()
print("None grads with grad checkpoint (no last; 1..n-1):", test(True, True, False))
print()
print("None grads with grad checkpoint (neither; 2..n-1):", test(True, False, False))
print()