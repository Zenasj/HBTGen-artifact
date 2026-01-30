import torch
import torch.nn as nn

torch.manual_seed(42)

x_cpu = torch.randn((5, 3, 10), device='cpu')
x_mps = x_cpu.detach().clone().to('mps')

model = nn.LSTM(10, 20, 1, batch_first=True)

h0_cpu = torch.zeros((1, 5, 20), device='cpu')
c0_cpu = torch.zeros((1, 5, 20), device='cpu')
h0_mps = h0_cpu.detach().clone().to('mps')
c0_mps = c0_cpu.detach().clone().to('mps')

out_cpu, _ = model(x_cpu, (h0_cpu, c0_cpu))
model.to('mps')
out_mps, _ = model(x_mps, (h0_mps, c0_mps))
print(f"{((out_cpu - out_mps.cpu()).abs() > 1e-7).sum() = }")
# Output:
# ((out_cpu - out_mps.cpu()).abs() > 1e-7).sum() = tensor(0)