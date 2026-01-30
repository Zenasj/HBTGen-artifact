import torch.nn as nn
import numpy as np

import torch
input = torch.tensor(1.234)
weight = torch.tensor(np.inf)
print(input, weight)  # tensor(1.2340) tensor(inf)
out = torch.nn.functional.prelu(input,weight)
print(out)  # tensor(1.2340)