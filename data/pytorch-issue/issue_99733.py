import torch

x = torch.ones((0, 6)).cuda()
lengths = torch.tensor([0, 0]).cuda()
torch.segment_reduce(x, "sum", lengths=lengths, unsafe=False, initial=0)