import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
x = torch.cuda.device(2)
print(x)
print(x.idx)