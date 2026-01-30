import torch
t1 = torch.Tensor([[.25,.1],[.25,.1],[.25,.1],[.25,.7]]).unsqueeze(2).permute(2,1,0)
t2 = t1.to('mps')
x = torch.distributions.Categorical(t1)
y = torch.distributions.Categorical(t2)
#samples normally, as expected
for i in range(10):
    print(x.sample())
#always returns [[2,2]]
for i in range(10):
    print(y.sample())