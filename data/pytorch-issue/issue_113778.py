import math
import operator
import torch


def func(x, y):
    # TODO: add boolean tests when SymBool is supported
    # to infer types
    return (
        torch.tensor([operator.add(x.item(), y.item())]),
        torch.tensor([operator.sub(x.item(), y.item())]),
        torch.tensor([operator.mul(x.item(), y.item())]),
        torch.tensor([operator.truediv(x.item(), y.item())]),
        torch.tensor([operator.floordiv(x.item(), y.item())]),
        torch.tensor([operator.pow(x.item(), y.item())]),
        torch.tensor([operator.abs(x.item())]),
        torch.tensor([operator.neg(x.item())]),
        torch.tensor([math.ceil(x.item())]),
        torch.tensor([math.floor(x.item())]),
    )

x = torch.randn(1, dtype=torch.float32)
y = torch.randn(1, dtype=torch.float32)

ep = torch.export.export(func, args=(x, y))
print(ep.graph_module.print_readable())