import torch.nn as nn
import random

#!/usr/bin/env python3
import argparse
import torch
torch.random.manual_seed(31337)

parser = argparse.ArgumentParser()
parser.add_argument("--bug", help="use buggy path", action="store_true")
parser.add_argument("--device", help="specify device", type=str, default="cpu")
args = parser.parse_args()

class Diff:
  def __init__(self, model, device, use_bug):
    self.model = model
    self.device = device
    self.use_bug = use_bug
    self.lossfn = torch.nn.NLLLoss(reduction="sum")
  def forward(self, x0_indices):
    x_indices = torch.zeros((1+1, x0_indices.shape[0], self.model.length), dtype=torch.long).to(self.device)
    q = torch.zeros((1+1, x0_indices.shape[0], self.model.length, 2)).to(self.device)
    x_indices[0,] = x0_indices
    q[1,] = 0.5*torch.ones(q[1,].shape)
    x_indices[1,] = torch.distributions.Categorical(q[1,]).sample()
    return x_indices, q
  def loss(self, x0_indices):
    x_indices, q = self.forward(x0_indices)
    if self.use_bug:
      pt = torch.log(torch.transpose(self.model(x_indices[1,], 1), -2, -1))
    else:
      pt = torch.transpose(torch.log(self.model(x_indices[1,], 1)), -2, -1)
    qt = torch.log(torch.transpose(q[1,], -2, -1))
    return self.lossfn(pt, x_indices[0,])

class MLP(torch.nn.Module):
  def __init__(self, length):
    super().__init__()
    self.length = length
    self.embed_input = torch.nn.Embedding(2, 50, padding_idx=0)
    self.readouts = torch.nn.Linear(50, 2)
    self.softmax = torch.nn.Softmax(dim=-1)
  def forward(self, x_indices, t):
    x = self.embed_input(x_indices)
    x = x.reshape((x.shape[0], self.length, -1))
    return self.softmax(self.readouts(x))

x0_indices = torch.zeros((200, 20))
for i in range(x0_indices.shape[0]):
  for j in range(i%5, x0_indices.shape[1], 5):
    x0_indices[i, j] = 1

model = MLP(x0_indices.shape[1]).to(args.device)
diff = Diff(model, args.device, args.bug)

optim = torch.optim.Adam(diff.model.parameters())
for epoch in range(10000):
  loss = diff.loss(x0_indices)
  print(f"[*] epoch={epoch}: loss={loss.item():.3f}")
  if loss < 0.0:
    print(f"[-] loss is not positive")
    break
  optim.zero_grad()
  loss.backward()
  optim.step()

import torch
print(f'Running PyTorch version: {torch.__version__}')

dtype = torch.float32

devices = [torch.device("mps"), torch.device("cpu")]

for device in devices:
    print(f"Using device: {device}")

    source = torch.randn(3, 1088, 2048, dtype=dtype, device=device)
    print("source: ", source.shape, source.dtype, source.device,
              source.cpu().numpy().flatten().min(), source.cpu().numpy().flatten().max())

    target = torch.clamp(torch.moveaxis(source, 0, -1), 0.0, 1.0)
    print("clamp(moveaxis(source)): ", target.shape, target.dtype, target.device,
              target.cpu().numpy().flatten().min(), target.cpu().numpy().flatten().max())

    target = torch.moveaxis(torch.clamp(source, 0.0, 1.0), 0, -1)
    print("moveaxis(clamp(source)): ", target.shape, target.dtype, target.device,
              target.cpu().numpy().flatten().min(), target.cpu().numpy().flatten().max())