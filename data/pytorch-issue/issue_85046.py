import torch

torch.bitwise_or(input=torch.ones([1,1]), other=torch.tensor([1]), out=torch.ones([2,1,4,13]))
torch.bitwise_xor(input=torch.ones([1,1]), other=torch.tensor([1]), out=torch.ones([2,1,4,13]))
torch.bitwise_and(input=torch.ones([1,1]), other=torch.tensor([1]), out=torch.ones([2,1,4,13]))
torch.bitwise_or(input=torch.tensor([-1]), other=torch.tensor([1]), out=torch.ones([1,0,1]))
torch.bitwise_xor(input=torch.tensor([-1]), other=torch.tensor([1]), out=torch.ones([1,0,1]))
torch.bitwise_and(input=torch.tensor([-1]), other=torch.tensor([1]), out=torch.ones([1,0,1]))

torch.logical_or(input=torch.ones([1,1]), other=torch.tensor([1]), out=torch.ones([2,1,4,13]))
torch.logical_xor(input=torch.ones([1,1]), other=torch.tensor([1]), out=torch.ones([2,1,4,13]))