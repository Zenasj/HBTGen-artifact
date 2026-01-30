import torch
a=torch.ones(1,dtype=torch.bfloat16)
print(torch.exp(a))

a=torch.ones(1,dtype=torch.float16)
print(torch.exp(a))