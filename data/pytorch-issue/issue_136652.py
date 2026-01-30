import torch
import torch.nn as nn

nt = torch.nested.nested_tensor([torch.randn(2, 3), torch.randn(3, 3)], layout=torch.jagged)
torch.manual_seed(5)
m2 = nn.Linear(3, 5)
compiled_m2 = torch.compile(m2)
out2 = compiled_m2(nt)
out2.to_padded_tensor(0).sum().backward()
print(compiled_m2._orig_mod.weight.grad)
# tensor([[ 0.9112, -0.5039, -3.5768],
#       [ 0.9112, -0.5039, -3.5768],
#        [ 0.9112, -0.5039, -3.5768],
#        [ 0.9112, -0.5039, -3.5768],
#        [ 0.9112, -0.5039, -3.5768]])
print(compiled_m2._orig_mod.bias.grad)
# None

import torch
import torch.nn as nn

nt = torch.nested.nested_tensor([torch.randn(2, 3), torch.randn(3, 3)], layout=torch.jagged)
torch.manual_seed(5)
m2 = nn.Linear(3, 5)
# compiled_m2 = torch.compile(m2)
out2 = m2(nt)
out2.to_padded_tensor(0).sum().backward()
print(m2.weight.grad)
# tensor([[-2.0948,  1.7890,  0.4688],
#        [-2.0948,  1.7890,  0.4688],
#        [-2.0948,  1.7890,  0.4688],
#       [-2.0948,  1.7890,  0.4688],
#        [-2.0948,  1.7890,  0.4688]])
print(m2.bias.grad)
# None