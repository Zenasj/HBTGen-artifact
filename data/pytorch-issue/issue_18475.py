import torch

def forward(self, x):
        u = x.mean(-1, keepdim=True)
        z = x - u
        s = (z * z).mean(-1, keepdim=True)
        x = z / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias