import torch
import torch._dynamo as dynamo
@dynamo.optimize("inductor")
def test_(a):
    c = 800.0
    return c / a

is_contiguous = True
a = torch.Tensor([427])
a.to(torch.float32)
n_iter = 5
for _ in range(n_iter):
    x = test_(a)
    print("x", x, x.item(), x.dtype)

In [4]: xx=torch.tensor(800/427)

In [5]: xx.item()
Out[5]: 1.8735363483428955

In [6]: b=torch.tensor(1/427)

In [8]: (b*800).item()
Out[8]: 1.873536229133606

In[1]: a = torch.tensor(1/427.0,dtype=torch.float)
In[2]: b = torch.tensor(800.0,dtype=torch.float)
In[3]: (a*b).item()
Out[3]: 1.873536229133606

In[1]: a = torch.tensor(427.0,dtype=torch.float)
In[2]: b = torch.tensor(800.0,dtype=torch.float)
In[3]: (b/a).item()
Out[3]: 1.8735363483428955