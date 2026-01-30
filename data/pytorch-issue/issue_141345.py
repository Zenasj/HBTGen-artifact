import torch
x = torch.tensor([complex(torch.nan, 0), complex(0, torch.nan), complex(torch.nan, torch.nan)], dtype=torch.complex64)
out = torch.asin(x)
out_gpu = torch.asin(x.cuda())
print("CPU Output:", out)
print("GPU Output:", out_gpu)