import torch

dtype = torch.half
device = torch.device("cuda")
m1 = torch.rand([128, 2400]).to(dtype).to(device).t()
m2 = torch.rand([2048, 25272]).to(dtype).to(device).t()[21940:24340]
bias = torch.rand([128]).to(dtype).to(device)
torch.addmm(bias, m2.t(), m1)