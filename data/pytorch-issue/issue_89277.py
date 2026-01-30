import torch
import torch.nn as nn

x_cpu = torch.arange(1, 5, dtype=torch.float32, device='cpu').view(1, 1, 2, 2)
x_mps = x_cpu.detach().clone().to('mps')

m = nn.Upsample(scale_factor=2, mode='nearest')
out1_cpu = m(x_cpu)
out1_mps = m(x_mps)
print(f"{((out1_cpu-out1_mps.cpu()).abs() > 1e-7).sum() = }")

m = nn.Upsample(scale_factor=2, mode='bilinear')
out2_cpu = m(x_cpu)
out2_mps = m(x_mps)
print(f"{((out2_cpu-out2_mps.cpu()).abs() > 1e-7).sum() = }")

m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
out3_cpu = m(x_cpu)
out3_mps = m(x_mps)
print(f"{((out3_cpu-out3_mps.cpu()).abs() > 1e-6).sum() = }")

input_3x3_cpu = torch.zeros(3, 3, device='cpu').view(1, 1, 3, 3)
input_3x3_cpu[:, :, :2, :2].copy_(x_cpu)
input_3x3_mps = torch.zeros(3, 3, device='mps').view(1, 1, 3, 3)
input_3x3_mps[:, :, :2, :2].copy_(x_mps)
print(f"{((input_3x3_cpu-input_3x3_mps.cpu()).abs() > 1e-7).sum() = }")

m = nn.Upsample(scale_factor=2, mode='bilinear')
out4_cpu = m(input_3x3_cpu)
out4_mps = m(input_3x3_mps)
print(f"{((out4_cpu-out4_mps.cpu()).abs() > 1e-7).sum() = }")

m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
out5_cpu = m(input_3x3_cpu)
out5_mps = m(input_3x3_mps)
print(f"{((out5_cpu-out5_mps.cpu()).abs() > 1e-6).sum() = }")

# Output:
# ((out1_cpu-out1_mps.cpu()).abs() > 1e-7).sum() = tensor(0)
# ((out2_cpu-out2_mps.cpu()).abs() > 1e-7).sum() = tensor(0)
# ((out3_cpu-out3_mps.cpu()).abs() > 1e-6).sum() = tensor(0)
# ((input_3x3_cpu-input_3x3_mps.cpu()).abs() > 1e-7).sum() = tensor(0)
# ((out4_cpu-out4_mps.cpu()).abs() > 1e-7).sum() = tensor(0)
# ((out5_cpu-out5_mps.cpu()).abs() > 1e-6).sum() = tensor(0)