import torch
device = torch.device("cuda:0")
t1 = torch.ones(1, device=device)

with torch.autograd.profiler.profile(use_cuda=True) as p:
    torch.add(t1, t1)

print(p.key_averages())
p.export_chrome_trace("/tmp/foo.txt")
print("Exported trace")