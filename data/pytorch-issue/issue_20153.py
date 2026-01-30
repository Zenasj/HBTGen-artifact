import torch

class BatchMod2(torch.jit.ScriptModule):
    def __init__(self, x_dim, z_dim):
        super().__init__()

    @torch.jit.script_method
    def forward(self, x, z):
        mean_scale = z.repeat(1, 2)
        mean, scale = torch.chunk(mean_scale[..., None, None], 2, 1)
        y = x * (scale + 1) + mean
        return y


b = BatchMod2(6, 6)
a1 = torch.ones(2,6,4,4)
a2 = torch.ones(2,6)
c=b(a1, a2)