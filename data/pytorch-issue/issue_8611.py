import torch

x = numpy.rand(3, 3)
t = torch.from_numpy(x)  # pylint complains that torch has no method from_numpy