py
import torch
from functorch.experimental.control_flow import cond
cell1 = torch.rand(3, 3)
cell2 = torch.rand(3, 3)
def test(x):
    def then():
        return cell1
    def els():
        return cell2
    return cond(x > 0, then, els, [])

opt_fn = torch._dynamo.optimize("eager")(test)
result1 = opt_fn(1)