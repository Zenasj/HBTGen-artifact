tensor([ 8.8750, -4.6875, -4.8750], dtype=torch.bfloat16)
tensor([1.1755e-38, 1.1755e-38, 1.1755e-38], dtype=torch.bfloat16)

tensor([0., -0., -0.], dtype=torch.bfloat16)

tensor([nan, nan, nan], dtype=torch.bfloat16)

import torch
import torch._refs as refs
import torch._prims as prims

a = torch.tensor([ 8.8750, -4.6875, -4.8750], dtype=torch.bfloat16)
b = torch.tensor([1.1755e-38, 1.1755e-38, 1.1755e-38], dtype=torch.bfloat16)

print(torch.fmod(a, b))
print(prims.fmod(a, b))
print(refs.fmod(a, b))