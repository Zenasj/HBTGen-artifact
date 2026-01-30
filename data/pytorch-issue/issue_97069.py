import torch

# Test fp32 tensor
tmp_tensor = torch.empty(1, dtype=torch.int8, device='cuda')
# memory usage: 1115MB (context)
tmp_tensor = torch.empty(1024*1024*1024, dtype=torch.float32, device='cuda')
# memory usage: 5213MB (increase about 4GB, right with tmp_tensor)

# Test fp16 tensor
del tmp_tensor
torch.cuda.empty_cache()
# memory usage: 115MB
tmp_tensor = torch.empty(1024*1024*1024, dtype=torch.float16, device='cuda')
# memory usage: 7621MB (increate about 6GB, wrong with the number tmp_tensor should be)