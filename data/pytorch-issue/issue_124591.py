import torch
import torch.nn as nn

import matplotlib.pyplot as plt

model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(3, 3, 1),
        )

x = torch.randn(2, 3, 10)
y = model(x).cpu().detach()

plt.plot(torch.linspace(0, 10, 20), y[0, 0], label='cpu')

y = model.to('mps')(x.to('mps')).cpu().detach()

plt.plot(torch.linspace(0, 10, 20), y[0, 0], label='mps')
plt.legend()
plt.show()

import matplotlib.pyplot as plt

model = nn.Sequential(
            nn.Conv1d(3, 3, 1),
            nn.Upsample(scale_factor=2),
        )

x = torch.randn(2, 3, 10)
y = model(x).cpu().detach()

plt.plot(torch.linspace(0, 10, 20), y[0, 0], label='cpu')

y = model.to('mps')(x.to('mps')).cpu().detach()

plt.plot(torch.linspace(0, 10, 20), y[0, 0], label='mps')
plt.legend()
plt.show()