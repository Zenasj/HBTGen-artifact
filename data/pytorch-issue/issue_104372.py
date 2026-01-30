import torch 
import torch._dynamo
from functorch.experimental.control_flow import cond

def test(pred, x):
    def true_fn(x):
        return x
    def false_fn(x):
        return -x

    return cond(pred, true_fn, false_fn, [x])

opt_test = torch.compile(test, backend="eager")
inp = torch.ones(3, 3)
assert not torch.allclose(opt_test(True, inp), opt_test(False, inp))