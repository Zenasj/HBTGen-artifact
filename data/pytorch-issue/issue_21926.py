import time
import torch
import numpy as np

a = torch.rand((40,100,256,256))
b = np.ones((40,100,256,256))
c = a.cpu()

t1 = time.time()
torch.save(a[0],open('torch1.p','wb'))
t2 = time.time()
print(t2 - t1)

a = a.cuda()
torch.save(a[0],open('torch2.p','wb'))
t3 = time.time()
print(t3 - t2)

np.save(open('numpy1.p','wb'),c[0])
t4 = time.time()
print(t4 - t3)

np.save(open('numpy2.p','wb'),b[0])
t5= time.time()
print(t5 - t4)

a = torch.randn(10, 20)
b = a[0]
b.add_(10) # changes `a`

a = torch.randn(10, 20)
b = a[0]
b.add_(10) # changes `a`

torch.save((a, b), 'foo.pt')
a, b = torch.save('foo.pt')

b.add_(10) # still changes `a`