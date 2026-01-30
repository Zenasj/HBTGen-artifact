import torch

def forward(self, x):
        """
        where x is a 2-D tensor
        """
        x[0,0] = 100
        y = torch.cat([x,x], dim = 0)
        return y