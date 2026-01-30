import torch.nn as nn

import torch
import torch.nn.functional as F

x = torch.tensor([], device="mps")
print(F.relu(x))