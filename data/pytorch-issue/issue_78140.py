import torch
torch.autograd.set_detect_anomaly(True)

x = torch.randn((100, 100), requires_grad=True)
y = torch.randn((100, 100), requires_grad=True)

mean_x = x.mean(dim=0, keepdim=True)
mean_y = y.mean(dim=0, keepdim=True)

offsets_x = x - mean_x
offsets_y = y - mean_y

cmd = (mean_x - mean_y).squeeze().norm(p=2)

central_moment_x = offsets_x.pow(2).mean(dim=0)
central_moment_y = offsets_y.pow(2).mean(dim=0)

cmd += (central_moment_x - central_moment_y).norm(p=2)

cmd.backward()