import torch
t = torch.ones(2,2,2).cuda()
t_backup = t.clone()
torch.irfft(t, 2, signal_sizes=(2,2))
print((t-t_backup).abs().sum().item()) # should be 0, but is 8