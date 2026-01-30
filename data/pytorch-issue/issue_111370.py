import torch
import numpy as np

def fn(value):
    x = np.array([0])
    # x = torch.as_tensor([0])    # this works
    return x.tolist()

opt_fn = torch.compile(fn)

x = np.array([1, 2, 3])
print("compiled: ", opt_fn(x))

import torch
import numpy as np

def fn():
    return np.typecodes["AllInteger"]

cnts = torch._dynamo.testing.CompileCounter()
opt_fn = torch._dynamo.optimize(cnts)(fn)

print(torch._numpy.typecodes['AllInteger'])
print(fn())
print(opt_fn())
print(cnts.frame_count)

Bbhil
bBhHiIlLqQpP
bBhHiIlLqQpP
0