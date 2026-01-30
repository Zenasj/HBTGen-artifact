import torch
import torch.nn as nn

kl_loss = nn.KLDivLoss(reduction="batchmean")

input = torch.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)

target = torch.softmax(torch.rand(3, 5), dim=1).double()

output = kl_loss(input, target)

output.backward()

# RuntimeError: Found dtype Float but expected Double