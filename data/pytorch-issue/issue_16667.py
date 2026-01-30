import torch

def _sparse_tensor_constructor(indices, values, size):
    return torch.sparse.FloatTensor(indices, values, size).coalesce()  # not coalesced from the default constructor

def __reduce__(self):
    return _sparse_tensor_constructor, (self._indices(), self._values(), self.size())

def _sparse_tensor_constructor(indices, values, size):
    return torch.sparse.FloatTensor(indices, values, size).coalesce()  # not coalesced from the default constructor 


def _reduce(x):
    # dispatch table cannot distinguish between torch.sparse.FloatTensor and torch.Tensor (?)
    if isinstance(x, torch.sparse.FloatTensor):
        return _sparse_tensor_constructor, (x._indices(), x._values(), x.size())
    else:
        return torch.Tensor.__reduce_ex__(x, pickle.HIGHEST_PROTOCOL)  # use your own protocol


class ExtendedPickler(pickle.Pickler):
    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[torch.Tensor] = _reduce
    # tried to use torch.sparse.FloatTensor instead of torch.Tensor but did not work (?)