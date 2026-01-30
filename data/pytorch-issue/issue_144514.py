import torch.nn as nn

import torch

# Load the shared library                                                                                                                               
torch.ops.load_library("build/libcustom_operator.so")

@torch.library.register_fake("cuequivariance_ops_torch::segmented_transpose_primitive")
def _(a: torch.Tensor, b: torch.Tensor, c: int) -> torch.Tensor:
    return torch.empty_like(a)

# Use the custom operator                                                                                                                               
a = torch.tensor([1.0, 2.0], device="cuda", requires_grad=True)
b = torch.tensor([3.0, 4.0], device="cuda", requires_grad=True)
c = 5

class Mul(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor, c: int) -> torch.Tensor:
        return torch.ops.cuequivariance_ops_torch.segmented_transpose_primitive(a, b, c);

from torch.library import opcheck
ret = opcheck(torch.ops.cuequivariance_ops_torch.segmented_transpose_primitive,(a, b, c))
print(ret)

mul = Mul()
# result = mul(a, b, c)                                                                                                                                 
# print(result)  # Output: tensor([20., 30.], device='cuda:0', grad_fn=<MulBackward1>)                                                                  

# mul = torch.export.export(mul, (a, b, c)).module()                                                                                                    
mul = torch.compile(mul)
result = mul(a, b, c)
print(result)  # Output:                                                                                                                                
result.sum().backward()
print(a.grad, b.grad)