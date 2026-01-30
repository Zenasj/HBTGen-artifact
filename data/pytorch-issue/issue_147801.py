import torch.nn as nn

import torch

# Test case to check hardswish_backward gradients at boundaries
input_values = torch.tensor([-3.0, 3.0, -2.0, 2.0], requires_grad=True)
out = torch.nn.functional.hardswish(input_values)
out.backward(torch.ones_like(input_values))

# Expected gradients:
# -3.0: should be 0 (flat region), but current code gives (x/3 +0.5) = -0.5
# 3.0: should be 1.0 (linear region), but current code gives 1.5
# -2.0: correct gradient (2*(-2)+3)/6 = -1/6 ≈ -0.1667
# 2.0: correct gradient (2*2+3)/6 = 7/6 ≈ 1.1667
expected_grad = torch.tensor([0.0, 1.0, -1/6, 7/6], dtype=torch.float32)

print(input_values.grad)
print(expected_grad)

tensor([-0.5000,  1.5000, -0.1667,  1.1667])
tensor([ 0.0000,  1.0000, -0.1667,  1.1667])