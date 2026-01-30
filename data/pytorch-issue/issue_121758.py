import torch.nn as nn

import torch
from torch.nested._internal.nested_tensor import ViewNestedFromBuffer, buffer_from_jagged

# set this to "False" and the test will stop failing
use_nt = True

with torch.profiler.profile() as prof:
    values = torch.randn(4, 6, requires_grad=True)
    offsets = torch.tensor([0,2,4])
    values = values * 2
    l = torch.nn.Linear(6,8)
    if use_nt:
        nt = ViewNestedFromBuffer.apply(values, offsets)
    else:
        nt = values

    nt = l(nt)
    if use_nt:
        val = buffer_from_jagged(nt)
    else:
        val = nt

    loss = val.sum()
    loss.backward()

found_seq_nr = False
for evt in prof.events():
    if "linear" in evt.name and "backward" not in evt.name:
        found_seq_nr = found_seq_nr or evt.sequence_nr != -1

assert found_seq_nr