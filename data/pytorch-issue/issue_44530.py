# torch.rand(B, 8, dtype=torch.float)
import torch
from torch.distributions import transforms as T

class CatTransformFixed(T.Transform):
    def __init__(self, tseq, dim=0, lengths=None, cache_size=0):
        assert all(t.event_dim == tseq[0].event_dim for t in tseq), "All transforms must have the same event_dim"
        super().__init__(cache_size=cache_size)
        self.transforms = tseq
        self.dim = dim
        self.lengths = lengths

    @property
    def event_dim(self):
        return self.transforms[0].event_dim

    def _call(self, x):
        parts = []
        start = 0
        for trans, length in zip(self.transforms, self.lengths):
            xslice = x.narrow(self.dim, start, length)
            parts.append(trans(xslice))
            start += length
        return torch.cat(parts, dim=self.dim)

    def log_abs_det_jacobian(self, x, y):
        logdetjacs = []
        start = 0
        for trans, length in zip(self.transforms, self.lengths):
            xslice = x.narrow(self.dim, start, length)
            yslice = y.narrow(self.dim, start, length)
            logdetjacs.append(trans.log_abs_det_jacobian(xslice, yslice))
            start += length

        dim = self.event_dim + (self.dim if self.dim < 0 else self.dim - x.dim())
        if dim < 0:
            return torch.cat(logdetjacs, dim=dim)
        else:
            return sum(logdetjacs)

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = T.AffineTransform(0., 1., event_dim=1)
        self.t2 = T.AffineTransform(0., 1., event_dim=1)
        self.tc = CatTransformFixed([self.t1, self.t2], dim=1, lengths=[4,4])

    def forward(self, x):
        return self.tc(x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 16  # Match batch size from original example
    return torch.rand(B, 8, dtype=torch.float)

