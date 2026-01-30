class TensorBase(Tensor):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        # workaround for https://github.com/pytorch/pytorch/issues/47091
        if len(types)>1:
            t = first(o for o in types if issubclass(o,TensorBase))
            if t: types = tuple(t if issubclass(o, TensorBase) else o for o in types)
        return super().__torch_function__(func, types, args=args, kwargs=kwargs)

import torch
from torch import _C

class MyTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        with _C.DisableTorchFunction():
            ret = func(*args, **kwargs)
            return _convert(ret, cls)

def _convert(ret, cls):
    if isinstance(ret, torch.Tensor):
        ret = ret.as_subclass(cls)

    if isinstance(ret, (tuple, list)):
        ret = type(ret)(_convert(r, cls) for r in ret)

    return ret

class MyTensor2(torch.Tensor):
    pass

if __name__ == "__main__":
    a = torch.tensor([5]).as_subclass(MyTensor)
    b = torch.tensor([6]).as_subclass(MyTensor2)

    print(type(a + b))  # MyTensor
    print(type(b + a))  # MyTensor