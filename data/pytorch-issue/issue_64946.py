import torch.nn as nn

import itertools
import torch
from torch.utils.benchmark import Timer

warmup = 2
device = "cuda"

datas = itertools.cycle([torch.randn(7, 10, device=device) for _ in range(10)])

class Model(torch.nn.Module):
    def forward(self, x):
        x.requires_grad_(True)
        y = x.square()
        return torch.autograd.grad(
            [y.sum()],
            [x],
        )[0]

model = Model()
model = model.to(device)
model = torch.jit.script(model)
model.eval()
model = torch.jit.freeze(model)

print("Warmup...")
for _ in range(warmup):
    model(next(datas).clone())

print("Starting...")
# just time
t = Timer(
    stmt="model(next(datas).clone())", globals={"model": model, "datas": datas}
)
perloop = t.timeit(n=10)

print(perloop)