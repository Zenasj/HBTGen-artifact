import torch.nn as nn

import torch
model = torch.nn.utils.spectral_norm(torch.nn.Linear(2, 5))
opt1 = torch.optim.SGD(model.parameters(), lr=1e-3)
opt2 = torch.optim.SGD(model.parameters(), lr=1e-3)

output = model(torch.randn(7, 2))
loss = output.abs().mean()

opt1.zero_grad(); loss.backward(retain_graph=True); opt1.step() # first propagation
opt2.zero_grad(); loss.backward(); opt2.step() # second