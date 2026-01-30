import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(False)

a = torch.Tensor([1e-4]).cuda().half()
a.requires_grad=True

# Clamp to avoid values that are too low
# Disable autograd while clamping because
# we only do this for numerical stability
with torch.no_grad():
    a.clamp_(min=1e-2)

a.norm(3).backward()
a.grad