import torch

@torch.jit.script
def f(p,a):
	return torch.erfinv(p) * a.sqrt()

p = torch.linspace(0.1,0.9,64,device="cuda")
#p = p.expand(320,-1).contiguous() #not a fix
a = torch.empty(320,1,device="cuda").uniform_(6., 15000.0)
a.requires_grad_() #no error in nograd mode
#p,a = torch.broadcast_tensors(p,a) #fix

r1=f(p,a)
r2=f(p,a)
print(torch.allclose(r1,r2))
print((r1-r2).abs().max())

import torch

@torch.jit.script
def f(p,a):
    tmp = a.sqrt()
    return (tmp,p * tmp)

p = torch.linspace(0.1,0.9,4,device="cuda")
#p = p.expand(320,-1).contiguous() #not a fix
a = torch.ones(16,1,device="cuda")
#a.requires_grad_() #no error in nograd mode
#p,a = torch.broadcast_tensors(p,a) #fix

r1=f(p,a)[1]
r2=f(p,a)[1]
print(torch.allclose(r1,r2))
print((r1-r2).abs().max())

@torch.jit.script
def f(p,a):
    tmp = a.sqrt()
    return (tmp,p + tmp)

p = torch.arange(1,65,dtype=torch.float32,device="cuda")
a = torch.arange(1,33,dtype=torch.float32,device="cuda").unsqueeze_(-1).mul_(100.0).square_()
a.requires_grad_() #no error in nograd mode

r1=f(p,a)[1]
r2=f(p,a)[1]
d = (r2-r1).abs()

print(torch.allclose(r1,r2))
print(d.shape, d.stride()) #32x64

for i in range(a.shape[0]):
    for j in range(p.shape[0]):
        if d[i,j] > 0.1: print(i,j, i * p.shape[0] + j, r1[i,j].item(), r2[i,j].item())