import torch.nn as nn

import torch

def print_par_shapes(parameters, prefix='\t'):
    i = -1
    for i, p in enumerate(pars):
        print('{}Parameter {}:'.format(prefix, i), p.shape)
    if i == -1:
        print('{}Iterator was empty'.format(prefix))

model = torch.nn.Linear(32, 32)
pars = model.parameters()
print('First iteration:')
print_par_shapes(pars)
print('Second iteration:')
print_par_shapes(pars)