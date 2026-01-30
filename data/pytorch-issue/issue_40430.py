import torch
device = torch.device("cuda")
torch.rand(10).to(device)