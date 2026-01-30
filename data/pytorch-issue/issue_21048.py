import torch
import torch.nn as nn

devices = xm.get_xla_supported_devices()

loss_fn = nn.NLLLoss()
model = XlaMNIST()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
for i in range(len(list(model.parameters()))):
    assert list(model.parameters())[i].device == optimizer.param_groups[0]["params"][i].device
model = model.to(devices[0])
for i in range(len(list(model.parameters()))):
    # This assert fails, because the model parameters have been moved to XLA device, while the parameters tracked by the optimizer are not moved.
    assert list(model.parameters())[i].device == optimizer.param_groups[0]["params"][i].device

data = torch.zeros(5, 1, 28,
                      28).to(devices[0])
target = torch.zeros(5, dtype=torch.int64).to(devices[0])

optimizer.zero_grad()
output = model(data)
loss = loss_fn(output, target)
loss.backward()
xm.optimizer_step(optimizer)