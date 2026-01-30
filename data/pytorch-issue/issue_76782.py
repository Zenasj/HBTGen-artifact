import torch.nn as nn

import torch
def fn(input):
    upper = True
    training = True
    fn_res = torch.nn.functional.rrelu(input, upper=upper, training=training)
    return fn_res

input = torch.rand([2], dtype=torch.float64, requires_grad=True)
res = fn(input)
print(res)
# tensor([0.7310, 0.9458], dtype=torch.float64,
#       grad_fn=<RreluWithNoiseBackward0>)
res.sum().backward()
# RuntimeError: Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.