import torch.nn as nn

import torch.nn.functional as f
import torch

# Precision error is calculated based on the Chebyshev distance.

args = torch.load('dropout.pt')

output = f.dropout(args['parameter:0'], args['parameter:1'], args['parameter:2'], args['parameter:3'])

args = torch.load('matmul.pt')

output = torch.matmul(output, args['parameter:1'])