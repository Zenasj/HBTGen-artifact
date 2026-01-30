import torch.nn as nn

foo1 = torch.from_numpy(np.array([0.25, 0.25, 0.25, 0.25]))
foo2 = torch.from_numpy(np.array([[0.25, 0.25, 0.25, 0.25]]))
foo3 = torch.from_numpy(np.array([[[0.25, 0.25, 0.25, 0.25]]]))

(array([26, 28, 46]), array([0, 1, 2, 3]))

(array([25, 28, 47]), array([0, 1, 2, 3]))

(array([100,   0,   0]), array([0, 1, 2, 3]))

import torch
import numpy as np
foo3 = torch.from_numpy(np.array([[[0.25, 0.25, 0.25, 0.25]]]))
torch.multinomial(foo3, 10, True)
# tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

out, hidden = lstm(input, hidden)
logits = w(out) # is nn.linear
choice = logits.softmax(dim=-1).multinomial() # goes pfft here because of dimension