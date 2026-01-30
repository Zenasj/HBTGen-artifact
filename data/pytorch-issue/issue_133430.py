import torch
device='mps'
a = torch.randn([32, 20064, 128], dtype=torch.float32,device=device)
b = torch.randn([32, 128, 20064], dtype=torch.float32, device=device)
res = torch.bmm(a, b)

print(pipeline.torch_dtype)
# torch.float32