import torch
import traceback
from torch import _dynamo as dynamo

def func(x, b=torch.tensor(1.0)):
    return x + b

x = torch.randn(3, 4)
b = torch.randn(3, 4)


# Trace without 'b'
# AssertionError: Dynamo input/output is not consistent with traced input/output
try:
    gm, guards = dynamo.export(func, x)
    gm.print_readable()

except AssertionError as e:
    traceback.print_exc()


# Trace with 'b'
# Succeed.
gm, guards = dynamo.export(func, x, b=b)
gm.print_readable()