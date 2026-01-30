import torch

def update(optimizer, x):
    optimizer.zero_grad()
    x.sum().backward()
    optimizer.step()

p = torch.zeros(10).requires_grad_(True).share_memory_() 
optim_multi = torch.optim.Adagrad([p]) 
optim_multi.share_memory() 
pool = torch.multiprocessing.Pool(1) 
pool.apply(update, (optim_multi, p))
print(optim_multi.state[p])

p = torch.zeros(10).requires_grad_(True)
optim_ref = torch.optim.Adagrad([p])
update(optim_ref, p)
print(optim_ref.state[p])

def share_memory(self):
    for group in self.param_groups:
        for p in group['params']:
            state = self.state[p]
            self.state[p] = multiprocessing.Manager().dict(state)