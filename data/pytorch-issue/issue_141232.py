import torch
from torch.utils._contextlib import _DecoratorContextManager
from torch.distributions import Normal, TransformedDistribution, AffineTransform, TanhTransform, ComposeTransform
from torchrl.modules import TanhNormal

torch.set_default_device("cpu")

if __name__ == "__main__":
    @torch.compile(fullgraph=True)
    def func(a):
        # 1. Breaks with Unsupported: Graph break due to unsupported builtin _C.expand
        d = TransformedDistribution(Normal(a, 1), ComposeTransform([TanhTransform(), AffineTransform(2, 2)]))
        b = d.log_prob(d.rsample((10,)))
        return b

        # 2. Breaks with call_method UserDefinedObjectVariable(instancemethod) __call__ [TensorVariable(), TupleVariable(length=2)] {}
        # return a.expand((10, 3))

        # 3. works
        # return torch.expand_copy(a, (10, 3))

        # 4. AttributeError: 'method_descriptor' object has no attribute '__module__'
        # expand = torch._C.TensorBase.expand
        # return expand(a, (10, 3))

    func(torch.randn(3))