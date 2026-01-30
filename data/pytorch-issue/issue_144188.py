import torch
f = torch.fill
cf = torch.compile(f)
input = torch.randn(1,2,3).to(torch.uint32).to('cuda')
value = -90
print(f(input,value))  # tensor([[[4294967206, 4294967206, 4294967206],[4294967206, 4294967206, 4294967206]]], device='cuda:0',dtype=torch.uint32)
cf_out = cf(input,value)