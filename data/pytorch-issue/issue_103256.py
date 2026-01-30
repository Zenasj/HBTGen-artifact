import torch
import torch.nn as nn

ckpt_path = './tmp/ckpt.pth'
model = torch.nn.Sequential(
    torch.nn.Linear(2, 3),
    torch.nn.Sigmoid(),
    torch.nn.Linear(3, 1),
    torch.nn.Sigmoid(),
)
input = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], device='cuda').reshape(3, 2)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), fused=True)

optimizer.zero_grad()
loss = model(input).sum()
loss.backward()
optimizer.step()

torch.save(
    {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    },
    ckpt_path,
)

# load
model = torch.nn.Sequential(
    torch.nn.Linear(2, 3),
    torch.nn.Sigmoid(),
    torch.nn.Linear(3, 1),
    torch.nn.Sigmoid(),
)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), fused=True)
checkpoint = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])
optimizer.load_state_dict(checkpoint["optimizer"])

optimizer.zero_grad()
loss = model(input).sum()
loss.backward()
optimizer.step()