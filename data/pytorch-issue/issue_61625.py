import torch

weight_fq = qconfig.weight() 
weight = torch.rand(1024, 512)
weight = Variable(weight, requires_grad=True)
o0 = weight_fq(weight)
o1 = weight_fq(weight)
o2 = weight_fq(weight)
o = o0 + o1 + o2
o.sum().backward()