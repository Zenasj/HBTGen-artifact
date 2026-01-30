import torch

def fn(x, index, src):
    x.index_put_([index], src, accumulate=True) # RuntimeError
    # o = torch.index_put(x, [index], src, accumulate=True) # RuntimeError
    # x[index] += src # works fine
    return x

x = torch.rand(3, 3)
src = torch.rand(3)
index = torch.tensor([1, 1, 1])

fn(x, index, src)
print('==== cpu eager mode OK! ====')

fn(x.cuda(), index.cuda(), src.cuda())
print('==== gpu eager mode OK! ====')