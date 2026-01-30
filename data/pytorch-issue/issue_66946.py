from math import log
import torch

class TorchLogUniform(torch.distributions.TransformedDistribution):
    def __init__(self, lb, ub):
        super(TorchLogUniform, self).__init__(
            torch.distributions.Uniform(lb.log(), ub.log()),
            torch.distributions.ExpTransform(),
        )

lu = TorchLogUniform(torch.tensor(1e8), torch.tensor(1e10))
lu.icdf(torch.tensor(0.1))