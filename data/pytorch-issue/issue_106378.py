# fails
import torch


class MyTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return super().__torch_function__(
            func,
            (torch.Tensor for _ in types),
            tuple(torch.Tensor() for _ in args),
            kwargs,
        )


def add(a, b):
    return torch.add(a, b)


t = MyTensor()
add(t, t)  # works
torch.compile(add, backend="eager")(t, t)  # RuntimeError

# works
import torch


class MyTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return torch.Tensor.__torch_function__(
            func,
            (torch.Tensor for _ in types),
            tuple(torch.Tensor() for _ in args),
            kwargs,
        )


def add(a, b):
    return torch.add(a, b)


t = MyTensor()
add(t, t)  # works
torch.compile(add, backend="eager")(t, t)  # works