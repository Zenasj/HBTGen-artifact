import numpy as np
import torch
import torch.nn as nn
loss = nn.KLDivLoss()
output = torch.from_numpy(np.array([[0.1132, 0.5477, 0.3390]])).float()
target = torch.from_numpy(np.array([[0.1541, 0.0511, 0.0947]])).float()
loss(output.log(),target)

import numpy as np
import torch
import torch.nn as nn
loss = nn.KLDivLoss()
output = torch.from_numpy(np.array([[0.1132, 0.5477, 0.3390]])).float()
target = torch.from_numpy(np.array([[0.1541, 0.0511, 0.0947]])).float()
loss(output.log(),target)

import torch
import torch.nn as nn

loss = nn.KLDivLoss()
output = torch.tensor([[0.1132, 0.5477, 0.3390]]).log_softmax(dim=1)
target = torch.tensor([[0.1541, 0.0511, 0.0947]]).softmax(dim=1)
print(loss(output, target))