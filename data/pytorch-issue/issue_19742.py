import torch

add11 = lambda: torch.Tensor([1]) + torch.Tensor([1])
add11() == add11()  # return tensor([True])
hash(add11()) == hash(add11())  # returns True
pickle.dumps(add11()) == pickle.dumps(add11())  # returns False
pickle.dumps(add11().data) == pickle.dumps(add11().data)  # returns False
pickle.dumps(torch.Tensor([1])) == pickle.dumps(torch.Tensor([1]))  # returns False

@functools.lru_cache(None)
def add(x, y):
 return x + y

for _ in range(10):
 tmp = add(torch.Tensor([1]), torch.Tensor([1]))
add.cache_info().hits  # returns 0