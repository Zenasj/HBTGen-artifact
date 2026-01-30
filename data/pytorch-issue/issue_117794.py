import torch
from torch.testing._internal.two_tensor import TwoTensorMode

def inp():
    base = torch.ones(2, 2)
    x = base.add(1)
    return x[:, 0]

# (2,)
print(inp().stride())

with TwoTensorMode():
    # (1,)
    print(inp().stride())

...

compiled_f = aot_function(
    f,
    fw_compiler=nop,
    bw_compiler=nop,
    decompositions=None,
    keep_inference_input_mutations=True,
    dynamic=False
)

with TwoTensorMode():
    tt_inp = inp()

compiled_f(tt_inp)