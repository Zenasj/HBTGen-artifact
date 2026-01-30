import torch.nn as nn

# add in the file torch/autograd/gradcheck.py
import torch
from torch.autograd import forward_ad

def get_fn():
    kernel_size = 2
    stride = 2
    return_indices = False
    arg_class = torch.nn.MaxPool1d(kernel_size, stride=stride, return_indices=return_indices)
    def fn(input):
        fn_res = arg_class(input)
        return fn_res
    return fn
fn = get_fn()
input_tensor = torch.rand([1, 1, 8], dtype=torch.float32)

input = input_tensor.clone().requires_grad_()
output = fn(input)
res = _get_analytical_jacobian_forward_ad(fn, (input,), (output,))
print(res)
# ((tensor([[0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.]]),),)


input = input_tensor.clone().requires_grad_()
tangents = []
for i in range(8):
    tensor = torch.zeros([1, 1, 8])
    tensor[0][0][i] = 1
    tangents.append(tensor)
with forward_ad.dual_level():
    for tangent in tangents:
        dual_input = forward_ad.make_dual(input, tangent)
        dual_output = fn(dual_input)
        column = forward_ad.unpack_dual(dual_output).tangent
        print(column)
# tensor([[[0., 0., 0., 0.]]])
# tensor([[[1., 0., 0., 0.]]])
# tensor([[[0., 1., 0., 0.]]])
# tensor([[[0., 0., 0., 0.]]])
# tensor([[[0., 0., 1., 0.]]])
# tensor([[[0., 0., 0., 0.]]])
# tensor([[[0., 0., 0., 1.]]])
# tensor([[[0., 0., 0., 0.]]])