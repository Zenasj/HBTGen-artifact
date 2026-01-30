import torch.nn as nn

import torch
a = torch.nn.Transformer.generate_square_subsequent_mask(3)
b = torch.nn.Transformer.generate_square_subsequent_mask(3,device='mps')
c = a.to(device='mps')
a == b.to('cpu')