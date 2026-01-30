import torch
torch.use_deterministic_algorithms(True)

a = torch.zeros((1,3,3,3)).cuda()
mask = torch.rand(a.shape) > 0.5
a[mask] = 1

a = a*(~mask) + mask

def forward(self, x):
        self.weight.data[self.mask] = 0
        return super(MaskedConv2d, self).forward(x)

def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)