import torch
from torch.autograd import forward_ad

input_tensor = torch.rand([5, 4, 5], dtype=torch.bfloat16)
exponent_tensor = torch.rand([1], dtype=torch.float64)

# backward
input = input_tensor.clone().requires_grad_()
exponent = exponent_tensor.clone()
torch.pow(input, exponent).sum().backward()
print("backward PASS")

# forward
input = input_tensor.clone().requires_grad_()
exponent = exponent_tensor.clone()
with forward_ad.dual_level():
    tangent = torch.rand_like(input)
    dual_input = forward_ad.make_dual(input, tangent)
    dual_output = torch.pow(dual_input, exponent)

# backward PASS
# RuntimeError: expected scalar type c10::BFloat16 but found double