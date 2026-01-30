import torch
import torch.nn as nn

NamedTuple = namedtuple("NamedTuple", ("a", "b"))

x = [(1, 2), NamedTuple(1, 2)]
y = torch.nn.parallel.scatter_gather.scatter(x, [0])
print(y)  # [[(1, 2), (1, 2)]]