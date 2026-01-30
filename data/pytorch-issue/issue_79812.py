import torch

device = torch.device('privateuseone:0')
cpu_tensor = torch.randn(2, 2)
priv_tensor = cpu_tensor.to(device)