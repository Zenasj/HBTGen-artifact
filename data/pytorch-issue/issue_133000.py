import torch.nn as nn

import torch.nn.functional as f
import torch

# Precision error is calculated based on the Chebyshev distance.

args = torch.load('gelu.pt')

output = f.gelu(args['parameter:0'])

args = torch.load('dropout.pt')

output = f.dropout(output, args['parameter:1'], args['parameter:2'], args['parameter:3'])

args = torch.load('linear.pt')

output = f.linear(output, args['parameter:1'], args['parameter:2'])