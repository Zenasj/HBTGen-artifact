import torch.nn as nn

import torch

asd = torch.nn.Parameter(torch.ones(16))

for i in range(2):
    print(f"Round {i}")
    with torch.no_grad():
        asd.set_(asd[1:])
        asd.grad=None

    m = torch.cat((asd, asd))
    m.sum().backward()