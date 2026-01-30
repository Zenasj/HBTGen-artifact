import torch as th
p = th.zeros((1024))
while True:
    if p.bernoulli().max() > 0:
        print('!')