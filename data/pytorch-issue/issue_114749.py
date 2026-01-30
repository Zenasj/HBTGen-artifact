import torch.nn as nn

import torch
print (f'torch version {torch.__version__}')
from torch import nn
device = 'cuda'

device ='cuda'

# Create a model. If I skip the ReLU, there is no error.
m1_non_compiled = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2)).to(device)

# Compile the model
m1_compiled = torch.compile(m1_non_compiled)

# create a tensor to pass through the network
X_test = torch.randn(2, 3)

print (f'device m1_non_compiled: {next(m1_non_compiled.parameters()).device}\n')
print (f'device m1_compiled: {next(m1_compiled.parameters()).device}')

# passing through compiled model results in an error
print (f'passed through non-compiled model\n{m1_non_compiled(X_test.to(device))}')

# passing through non-compiled model works
print (f'passed through compiled model\n{m1_compiled(X_test.to(device))}')