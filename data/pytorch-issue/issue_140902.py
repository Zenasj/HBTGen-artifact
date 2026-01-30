import torch
import torch.nn as nn

device = torch.device('mps')

INPUT_CHANNELS = 100
OUTPUT_CHANNELS = 50
FEATURES = 25

conv = nn.Conv1d(INPUT_CHANNELS, OUTPUT_CHANNELS, kernel_size=3, padding=1).to(device)
norm = nn.LayerNorm(OUTPUT_CHANNELS).to(device)

a = torch.rand((1, INPUT_CHANNELS, FEATURES)).to(device)
b = conv(a)
c = b.transpose(1, 2).contiguous()
d = norm(c)
e = d.transpose(1, 2).contiguous()
loss = torch.sum(e)
print([var.is_contiguous() for var in (a, b, c, d, e, loss)])
loss.backward()

device,ic,oc,f = torch.device('mps'), 1, 2, 3

bias = torch.rand(oc, device=device, requires_grad=True)
weight = torch.rand(oc, ic, 3, device=device, requires_grad=True)
out = torch.nn.functional.conv1d(torch.rand(1, ic, f, device=device), weight, bias, padding=1)
torch.autograd.grad((out,), (weight, bias), (torch.rand(1, f, oc, device=device).transpose(1, 2),))