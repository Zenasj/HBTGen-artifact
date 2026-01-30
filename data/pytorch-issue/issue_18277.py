import torch

shape = (1000, 1000)
for i  in range(10000):
    if i > 9990: 
        print(tr.ones(*shape).fill_(i).mean(), "   \t",  tr.ones(*shape).fill_(i).cuda().mean())

print(tr.ones(*shape, dtype=torch.float64).fill_(i).mean(), "   \t",  tr.ones(*shape, dtype=torch.float64).fill_(i).cuda().mean())