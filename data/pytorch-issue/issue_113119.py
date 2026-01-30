import torch

def forward(n,x):     
  return torch._C._special.special_hermite_polynomial_he(n=n, x=x)

n = torch.rand([4], dtype=torch.float32)
x = torch.rand([5, 1, 1], dtype=torch.float16)
no_op_info = forward(n,x)# on eagermode
print("build succeeded")
op_info = torch.compile(forward)(n,x)# on torch.compile mode