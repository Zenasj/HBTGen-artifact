import torch.nn as nn

torch.nn.functional.one_hot(torch.arange(0, t, dtype=torch.int64))

import torch
from test_utils import cpu
torch._dynamo.config.specialize_int=False
def test_one_hot():
    input_shapes = [
        (16),
        (10),
        (11),
        (4)
    ]
    def wrapper_fn(t):
        t2 = torch.nn.functional.one_hot(torch.arange(0, t, dtype=torch.int64))
        return t2

    f_cpu = torch.compile(wrapper_fn, dynamic=True)

    for shape in input_shapes:
        y_cpu = f_cpu(shape)