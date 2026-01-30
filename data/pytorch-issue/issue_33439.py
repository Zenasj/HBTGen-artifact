import torch
x=torch.zeros(3,3)
x[1][1]=1
x=x.to_sparse()
torch.save(x, './scratch/test4e')
y = torch.load('./scratch/test4e')