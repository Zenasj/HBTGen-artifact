import torch

torch.add(input=torch.ones([2,2]), other=1, out=torch.ones([2,1,1,1], dtype=torch.float64))
torch.add(input=torch.tensor([1]), other=torch.ones([1,1,1]), out=torch.ones([1,5,1,1,1]))

torch.ne(input=torch.ones([2,2]), other=1, out=torch.ones([2,1,1,1], dtype=torch.float64))
torch.ge(input=torch.ones([2,2]), other=1, out=torch.ones([2,1,1,1], dtype=torch.float64))
torch.gt(input=torch.ones([2,2]), other=1, out=torch.ones([2,1,1,1], dtype=torch.float64))
torch.le(input=torch.ones([2,2]), other=1, out=torch.ones([2,1,1,1], dtype=torch.float64))
torch.lt(input=torch.ones([2,2]), other=1, out=torch.ones([2,1,1,1], dtype=torch.float64))
torch.eq(input=torch.ones([2,2]), other=1, out=torch.ones([2,1,1,1], dtype=torch.float64))

torch.ne(input=torch.ones([2,2]), other=1, out=torch.ones([2,1,1,1,1]))