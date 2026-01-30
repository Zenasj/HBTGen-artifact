import torch
from torch.autograd import forward_ad

input_tensor = torch.rand([2, 2], dtype=torch.float32)
other_tensor = torch.rand([2, 2], dtype=torch.float32)

# backward
input = input_tensor.clone().requires_grad_()
other = other_tensor.clone().requires_grad_()
torch.pow(input, other).sum().backward()
print("backward PASS")

# forward
input = input_tensor.clone().requires_grad_()
other = other_tensor.clone().requires_grad_()
with forward_ad.dual_level():
    tangent1 = torch.rand_like(input)
    dual_input = forward_ad.make_dual(input, tangent1)

    tangent2 = torch.rand_like(other)
    dual_other = forward_ad.make_dual(other, tangent2)
    dual_output = torch.min(dual_input, dual_other)
    # dual_output = torch.max(dual_input, dual_other)

# backward PASS
# RuntimeError: expected scalar type double but found float

import torch
from torch.autograd import forward_ad

input_tensor = torch.rand([2, 2], dtype=torch.float32)
other_tensor = torch.rand([2, 2], dtype=torch.float32)

# backward
input = input_tensor.clone().requires_grad_()
other = other_tensor.clone().requires_grad_()
torch.pow(input, other).sum().backward()
print("backward PASS")

# forward
input = input_tensor.clone().requires_grad_()
other = other_tensor.clone().requires_grad_()
with forward_ad.dual_level():
    tangent1 = torch.rand_like(input)
    dual_input = forward_ad.make_dual(input, tangent1)

    tangent2 = torch.rand_like(other)
    dual_other = forward_ad.make_dual(other, tangent2)
    dual_output = torch.fmin(dual_input, dual_other)
    print("forward PASS")
    print(dual_output)

# backward PASS
# forward PASS
# tensor([[8.6834e-01, 4.9130e-01],
#         [6.5345e-04, 3.8819e-01]], grad_fn=<FminBackward0>,
#        tangent=tensor([[0.2419, 0.3841],
#         [0.6100, 0.2944]]))