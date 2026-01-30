import torch.nn as nn

cpp
import torch
model = torch.nn.Softplus(threshold=1).double()
input = torch.tensor(0.9, dtype=torch.double, requires_grad=True)
output = model(input)
print(input.item(), output.item())
torch.autograd.gradcheck(model, input)