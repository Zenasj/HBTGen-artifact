import torch

t1 = torch.Tensor()

for i in range(2):
    t_temp = torch.Tensor([i])
    t1 = torch.cat((t1, t_temp), dim=0)

t2 = torch.Tensor().to("mps")

for i in range(2):
    t_temp = torch.Tensor([i]).to("mps")
    t2 = torch.cat((t2, t_temp), dim=0)