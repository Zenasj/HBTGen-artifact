import torch

class MyMapping(object):
    def __init__(self, d):
        self._d = d

    def __getitem__(self, key):
        try:
            value = self._d[key]
        except KeyError:
            raise KeyError(key) from None
        return value

d = MyMapping({"a": 10, "b": 20})

def mapping_get(obj, key, value=None):
    try:
        return obj.__getitem__(key)
    except KeyError:
        return value

@torch.compile(backend="eager", fullgraph=True)
def fn(x, d, key):
    x = torch.sin(x + 1)
    return x, mapping_get(d, key)


x = torch.rand(2, 3)
print(fn(x, d, "m"))