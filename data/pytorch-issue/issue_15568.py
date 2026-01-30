import torch

n = int(3e4)

d0 = torch.device('cuda:0')
x0 = torch.zeros(n, n, device=d0)

d1 = torch.device('cuda:1')
x1 = torch.zeros(n, n, device=d1)
x1 + 1
x1 + 1


with torch.cuda.device(d1):
    s = torch.cuda.Stream()
with torch.cuda.stream(s):
    x2 = x0.to(d1)

print(x2[0, 0])

import torch

n = int(3e4)    

d0 = torch.device('cuda:0')
x0 = torch.zeros(n, n, device=d0)

d1 = torch.device('cuda:1')
x1 = torch.zeros(n, n, device=d1)

d2 = torch.device('cuda:2')
x2 = torch.zeros(n, n, device=d2)

s0 = torch.cuda.Stream()
s01 = torch.cuda.Stream()
s02 = torch.cuda.Stream()

torch.cuda.synchronize()
with torch.cuda.device(d1):
	torch.cuda.synchronize()
with torch.cuda.device(d2):
	torch.cuda.synchronize()


with torch.cuda.stream(s0):
	x0 @ x0

with torch.cuda.stream(s01):
    x1 = x0.to(d1)

with torch.cuda.stream(s02):
    x2 = x0.to(d2)       

torch.cuda.synchronize()
with torch.cuda.device(d1):
	torch.cuda.synchronize()
with torch.cuda.device(d2):
	torch.cuda.synchronize()

s01 = torch.cuda.Stream(d1)
s02 = torch.cuda.Stream(d2)

with torch.cuda.device(d1):
	with torch.cuda.stream(s01):
	    x1 = x0.to(d1)

s1 = torch.cuda.Stream(d1)
with torch.cuda.device(d1):
    with torch.cuda.stream(s1):
        x1 + 1
        x1 + 1

with torch.cuda.device(d1):
    with torch.cuda.stream(default_stream1):
        x2 = x2.t() @ x2