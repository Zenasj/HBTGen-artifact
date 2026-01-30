import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query_conv(x).contiguous().reshape(B, -1, H*W)
        key   = self.key_conv(x).contiguous().reshape(B, -1, H*W)
        value = self.value_conv(x).contiguous().reshape(B, -1, H*W)

        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = torch.softmax(attention, dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.contiguous().reshape(B, C, H, W)
        return self.gamma * out + x

class SimpleDenoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.attn = SelfAttentionBlock(32)
        self.conv_out = nn.Conv2d(32, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv_in(x)
        x = self.attn(x)
        x = self.conv_out(x)
        return x

model = SimpleDenoiseNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

input_data = torch.rand(2, 3, 32, 32, device=device)
target_data = torch.rand(2, 3, 32, 32, device=device)

optimizer.zero_grad()
output = model(input_data)
loss = criterion(output, target_data)
loss.backward()  # Fails on MPS, works on CPU/CUDA
optimizer.step()

import torch
device,ic,oc,f = 'mps', 1, 2, 3

bias = torch.rand(oc, device=device, requires_grad=True)
weight = torch.rand(oc, ic, 3, device=device, requires_grad=True)
inp = torch.rand(1, ic, f, device=device, requires_grad=True)
out = torch.nn.functional.conv1d(inp, weight, bias, padding=1)
torch.autograd.grad((out,), (inp, weight, bias), (torch.rand(1, f, oc, device=device).transpose(1, 2),))