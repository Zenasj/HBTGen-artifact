import torch.nn as nn

import torch
x_fp32 = torch.tensor([-1,2,4,7], requires_grad=True, dtype=torch.float32, device="cuda")
x_bf16 = torch.tensor([-1,2,4,7], requires_grad=True, dtype=torch.bfloat16, device="cuda")
torch.nn.functional.relu6(x_fp32).sum().backward()
torch.nn.functional.relu6(x_bf16).sum().backward()
assert (x_fp32.grad == x_bf16.grad).all()