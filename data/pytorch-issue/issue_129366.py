layout = torch.jagged
g0 = torch.zeros(1)
g1 = torch.zeros(2)
g = torch.nested.nested_tensor([g0, g1], layout=layout)
torch.save(g, "file.p")

import torch
from torch.nested._internal.nested_tensor import nested_view_from_values_offsets, _nt_view_dummy

def fn(values, offsets):
    return nested_view_from_values_offsets(values, offsets)

fn_t = torch.fx.symbolic_trace(fn)

# this is needed to call sym_sizes_capsule, presumably:
# https://github.com/pytorch/pytorch/blob/d95a019704f526ee985cfe8d68261d34ddaf0e9d/torch/csrc/PyInterpreter.cpp#L812-L813
_nt_view_dummy().size()

torch.save(fn_t, "/tmp/py_capsule_repro.pt")

import torch

layout = torch.jagged
g0 = torch.zeros(1)
g1 = torch.zeros(2)
g = torch.nested.nested_tensor([g0, g1], layout=layout)
# accessing size caches a PyCapsule on the object
g.shape
# TypeError: cannot pickle 'PyCapsule' object
torch.save(g, "file.p")