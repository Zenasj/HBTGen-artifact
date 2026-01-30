import torch.nn as nn

import torch
import torch.nn.functional as F

device = torch.device('mps')
logits = torch.tensor([0.5749, -0.0438, -0.0700, -0.1062, -0.0332, -0.0981, -0.0614, -0.0751, -0.0396, -0.0733], device=device)
pred = F.softmax(logits).argmax()

print(pred)