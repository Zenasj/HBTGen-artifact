import torch.nn as nn

import numpy
import torch as th
from collections import OrderedDict

#dev = th.device("cuda")
dev = th.device("cpu")
dtype = th.float32

model = th.nn.Sequential(OrderedDict((
    ("input", th.nn.Linear(14650, 2)),
    ("output_activ", th.nn.Softmax(dim=-1))
)))

model.to(dev, dtype=dtype)
model.train()

optim = th.optim.Adam(model.parameters(), lr=1e-5)
loss = th.nn.BCELoss()

inputs = th.tensor(numpy.load("input.npy"), device=dev, dtype=dtype)
outputs = th.tensor(numpy.load("output.npy"), device=dev, dtype=dtype)

print("inputs = ", inputs)
print("outputs = ", outputs)

optim.zero_grad()
m = model(inputs)
lll = loss(m, outputs)
lll.backward()
print("Gradients:")
for p in model.parameters():
    if p.grad is None:
        continue
    grad = p.grad.data
    print(grad)

inputs = th.randn(14650, device=dev, dtype=dtype)