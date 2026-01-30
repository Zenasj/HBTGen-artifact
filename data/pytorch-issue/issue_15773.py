import torch

a,b = torch.ones(3,4,1,5,requires_grad=True),torch.ones(3,4,1,5,requires_grad=True)
s = torch.distributions.beta.Beta(a,b).rsample(torch.Size((10,)))
torch.sum(s).backward()
a.grad

a,b = torch.ones(3,4,1,5,requires_grad=True,device='cuda'),torch.ones(3,4,1,5,requires_grad=True,device='cuda')
s = torch.distributions.beta.Beta(a,b).rsample(torch.Size((10,)))
torch.sum(s).backward()
a.grad

a,b = torch.ones(3,4,1,5,requires_grad=True,device='cuda'),torch.ones(3,4,1,5,requires_grad=True,device='cuda')

a_cpu = a.cpu()
b_cpu = b.cpu()

s_cpu = torch.distributions.beta.Beta(a_cpu,b_cpu).rsample(torch.Size((10,)))
s = s_cpu.cuda()
torch.sum(s).backward()
a.grad