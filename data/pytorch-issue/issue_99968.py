import torch
import torch.nn as nn

net = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
)

with torch.no_grad():
  for param in net.parameters():
    for j in param.flatten():
        #print("current j", j)
        j += 1

a = torch.rand(1, requires_grad=True)

with torch.no_grad():
    b = a[:]
    b += 1

# Doing any of the of the following produces an error
b.sin()   # (1) 
b.grad_fn  # (2)
print(b)  # (3) the reason this fails is because it calls into t.grad_fn for printing purposes