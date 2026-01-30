import torch

# var_a in cuda, var_b in CPU
a = torch.randn((2, 2), device=torch.device('cuda'))
b = torch.randn((2, 3), device=torch.device('cpu'))

# matrix multiplication is fine
c = a @ b
d = torch.matmul(a, b)

# use torch built-in function to create tensor on CUDA is also fine
good_var = torch.randn((1, 10), device=torch.device('cuda'))

# but the following operations will failed
bad_var = torch.tensor((1, 10), device=torch.device('cuda'))

import torch

# var_a in cuda, var_b in cpu
a = torch.randn((2, 2), device=torch.device('cuda'))
b = torch.randn((2, 3), device=torch.device('cpu'))

# matrix multiplication is fine
c = a @ b
d = torch.matmul(a, b)

# use torch built-in function to create tensor on cuda is also fine
good_var = torch.randn((1, 10), device=torch.device('cuda'))

# but the following operations will failed
# bad_var = torch.tensor((1, 10), device=torch.device('cuda'))

# also won't be able to asses var_a against
print(a)