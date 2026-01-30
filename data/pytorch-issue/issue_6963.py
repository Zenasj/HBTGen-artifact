import torch
import torch.nn as nn
from torch.autograd import Variable
crit = nn.MSELoss()

####################
torch.manual_seed(876)
a = Variable(torch.randn(3,64,64), requires_grad=True)

torch.manual_seed(875)
b = Variable(torch.randn(3,64,64))
c = crit(a,b)
#####################
torch.manual_seed(874)
d = Variable(torch.randn(3,64,64), requires_grad=True)

torch.manual_seed(873)
e = Variable(torch.randn(3,64,64))
f = crit(d,e)

####################
torch.manual_seed(876)
h = Variable(torch.randn(3,64,64), requires_grad=True)

torch.manual_seed(875)
i = Variable(torch.randn(3,64,64))
j = crit(h,i)
#####################

test = (j, f, c)

g = 0
for loss in test: 
   g+=loss 
g.backward(retain_graph=True)


print("Rounded Results:")
print("Using += before backward(): " + str(float(g)))
print("Using a single float(): " + str( float(j + f + c)  ))
print("Using a single item(): " + str( (j + f + c).item()  ))

print("Working Methods:")
print("Adding j, f, c, separately using float(): " + str(float(j) + float(f) + float(c) ))
print("Adding j, f, c, separately using item(): " + str(j.item() + f.item() + c.item() ))

print("The difference Between Adding Methods:")
print("Difference: " + str( (float(j) + float(f) + float(c)) - float(g) ))

summed = torch.sum(torch.cat([j.unsqueeze(0), f.unsqueeze(0), c.unsqueeze(0)])).item()
(g - summed).item()

-0.1887800004
-0.19992000004
-0.00301999994
0.02676000003

totalLoss += float(mod.loss)

totalLoss += mod.loss