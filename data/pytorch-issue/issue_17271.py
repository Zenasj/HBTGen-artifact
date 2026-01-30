import torch
print(torch.__version__) #1.0.0.dev20190207
a=torch.ones(2**31,dtype=torch.float32)
print(a)             #tensor([1., 1., 1.,  ..., 1., 1., 1.])
print(torch.sign(a)) #tensor([1., 1., 1.,  ..., 1., 1., 1.])
print(torch.exp(a))  #tensor([0., 0., 0.,  ..., 0., 0., 0.])  THIS IS WRONG