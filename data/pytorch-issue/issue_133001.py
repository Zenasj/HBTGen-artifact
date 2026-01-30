import torch.nn as nn

import torch.nn.functional as f
import torch

# Precision error is calculated based on the Chebyshev distance.

args = torch.load('batch_norm.pt')

output = f.batch_norm(args['parameter:0'], args['parameter:1'], args['parameter:2'], args['parameter:3'], args['parameter:4'], args['parameter:5'], args['parameter:6'], args['parameter:7'])

args = torch.load('relu.pt')

output = f.relu(output)

args = torch.load('max_pool2d.pt')

output = f.max_pool2d(output, args['parameter:1'], args['parameter:2'], args['parameter:3'], args['parameter:4'])

args = torch.load('conv2d.pt')

output = f.conv2d(output, args['parameter:1'], args['parameter:2'], args['parameter:3'], args['parameter:4'], args['parameter:5'], args['parameter:6'])