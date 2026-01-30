import torch

rel = (3.4028234663852886e+38 - 3.3895313892515355e+38)/3.4028234663852886e+38
print("REL:", rel)
print(torch.finfo(torch.bfloat16))
print(torch.finfo(torch.float32))
print(torch.tensor(3.4028234663852886e+38, dtype=torch.bfloat16))