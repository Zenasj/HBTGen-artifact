import torch.nn as nn

import torch.nn.functional as f
import torch

# Precision error is calculated based on the Chebyshev distance.

args = torch.load('conv2d.pt')

output = f.conv2d(args['parameter:0'], args['parameter:1'], args['parameter:2'], args['parameter:3'], args['parameter:4'], args['parameter:5'], args['parameter:6'])

args = torch.load('batch_norm.pt')

output = f.batch_norm(output, args['parameter:1'], args['parameter:2'], args['parameter:3'], args['parameter:4'], args['parameter:5'], args['parameter:6'], args['parameter:7'])