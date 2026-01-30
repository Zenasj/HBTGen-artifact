import torch
import numpy as np


def model(mu):
    aa = [3, 4, 77, 9]
    aa = np.asarray(aa)
    aa_ts = torch.Tensor(aa).cuda().float()
    mu = mu + aa_ts
    # need to reset some idx of the valve but it not work in trace.jit
    mu[1] = 55
    return mu


x1 = torch.tensor([99, 2, 3, 4]).cuda().float()
x2 = torch.tensor([0, 0, 0, 0]).cuda().float()

print(model(x1))
print(model(x2))

fn = torch.jit.trace(model, x1)
fn.save("test_model_freeze.pt")
model_new = torch.jit.load('test_model_freeze.pt')
y1 = model_new(x1)
y2 = model_new(x2)
print(y1)
print(y2)