import torch

x = torch.zeros(1, 1, 1, 1, 1, device="mps")
x.all()       # crash
x.all(dim=0)  # crash
x.all(dim=1)  # no crash
x.all(dim=2)  # no crash
x.all(dim=3)  # no crash
x.all(dim=4)  # no crash