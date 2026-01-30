import torch
from torch.autograd import forward_ad

input = torch.randint(0, 2, [5], dtype=torch.bool)
other_tensor = torch.rand([5], dtype=torch.float32)

# backward
other = other_tensor.clone().requires_grad_()
torch.fmax(input, other).sum().backward()
print("backward PASS")

# forward
other = other_tensor.clone().requires_grad_()
with forward_ad.dual_level():
    tangent = torch.rand_like(other)
    dual_other = forward_ad.make_dual(other, tangent)
    dual_output = torch.fmin(input, dual_other)
    print("forward PASS")
    print(dual_output)

# backward PASS
# RuntimeError: Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.