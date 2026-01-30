import torch
from torch._dynamo import compiled_autograd
compiled_autograd.compiled_autograd_enabled = True
compiled_autograd.reset()
assert(compiled_autograd.compiled_autograd_enabled == False)