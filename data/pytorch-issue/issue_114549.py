import torch
import torch.nn as nn
torch.backends.cudnn.deterministic = True

device = 'cuda'
M = 192
B = 3
H = 36
W = 48
a = torch.randn(B, 4 * M, H, W).to(device)

entropy_parameters = nn.Conv2d(in_channels=M * 4, out_channels=M * 2, kernel_size=1, stride=1, padding=0).to(device).eval()

b = entropy_parameters(a)  # direct forward

c = torch.zeros_like(b)  # for loop
for k in range(B):
    for h in range(H):
        for w in range(W):
            c[k:k + 1, :, h:h + 1, w:w + 1] = entropy_parameters(a[k:k + 1, :, h:h + 1, w:w + 1])

d = (b - c == 0).all()
error = b - c
g = torch.allclose(b, c)
print(d)
print(g)