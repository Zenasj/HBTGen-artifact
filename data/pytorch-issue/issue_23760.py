import torch.nn as nn

import torch


class Model(torch.nn.Module):

    def forward(self, a):
        return a


device = torch.device('cuda')
model = Model()
model.to(device)
a = torch.randint(0, 255, (256, 900, 2), device=device)
print('`torch.nn.parallel.data_parallel`')
torch.nn.parallel.data_parallel(module=model, inputs=(a,))