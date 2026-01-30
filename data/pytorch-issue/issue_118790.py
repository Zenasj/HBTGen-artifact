# # In case of dtype promotion, faketensor converted into tensor.
    # # Need to convert into faketensor if input was a faketensor.
    # if dtype_converted:
    #     x = wrap_output_with_input_device_(x, common_device)

import torch
def fn(a):
 b = a.t()
 b.mul_(1.0)
 return b

x = torch.arange(6).reshape([2, 3]).to('cpu')

print("x ", x.cpu())

compiled_fn = torch.compile(fn)
y = compiled_fn(x)

print("y ", y.cpu())