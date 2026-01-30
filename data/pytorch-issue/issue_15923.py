import time
import torch
import numpy as np
from torch.autograd import Variable


n_actions = 22
dt = 0.01

jac_t = torch.randn(6, n_actions)
state = torch.randn(6, 1)
target = torch.matmul(jac_t, torch.randn(n_actions, 1)) * dt + state

print("target:", target)

t0 = time.perf_counter()
q_dot = Variable(torch.randn(n_actions, 1), requires_grad=True)
v = [q_dot]
optimizer = torch.optim.LBFGS(v)#, lr=0.1)
for i in range(0, 10):
    def cost():
        optimizer.zero_grad()
        next_state = torch.matmul(jac_t, q_dot) * dt + state
        d = torch.pow(next_state - target, 2).sum()
        d.backward()
        return d
    optimizer.step(cost)
    d = cost()
    if d < 1e-3:
        break
print(time.perf_counter() - t0, " s")