import torch
device = torch.device("cuda:0")
aa = torch.ones(60,60).to(device)
torch.fft.fft2(aa)