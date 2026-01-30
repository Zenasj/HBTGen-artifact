import torch
a = torch.zeros(2)
b = a.expand(2, 2)

b *= 2 # breaks
# RuntimeError: mul: unsupported operation: more than one element of the written-to tensor refers to a single memory location. Please clone() the tensor before calling mul

mask = torch.ByteTensor([[0, 1], [1, 0]])
b.masked_fill_(mask, 0) # works, but semantically unsafe