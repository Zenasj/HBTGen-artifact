import torch.nn as nn

import torch.nn.functional as f
import torch

# Precision error is calculated based on the Chebyshev distance.

args = torch.load('__add__.pt')

output = torch.Tensor.__add__(args['parameter:0'], args['parameter:1'])

args = torch.load('layer_norm.pt')

output = f.layer_norm(output, args['parameter:1'], args['parameter:2'], args['parameter:3'], args['parameter:4'])

args = torch.load('__add__2.pt')

output = torch.Tensor.__add__(output, args['parameter:1'])