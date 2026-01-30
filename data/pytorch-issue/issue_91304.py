import torch.nn as nn

import torch
from torch.nn.utils import weight_norm
from torch.nn.utils import remove_weight_norm

truth_order = torch.nn.Linear(1, 1).state_dict()
new_order = remove_weight_norm(weight_norm(torch.nn.Linear(1, 1))).state_dict()

print(truth_order.keys())
print(new_order.keys())