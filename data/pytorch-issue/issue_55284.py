# torch.rand(1, dtype=torch.float32)
import torch
import logging

class LoggingTensor(torch.Tensor):
    @classmethod
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        torch.Tensor._make_subclass(instance, instance, True)
        return instance

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # Avoid recursion by excluding __repr__ and __str__
        if func.__name__ in ('__repr__', '__str__'):
            return super().__torch_function__(func, types, args, kwargs)
        if kwargs is None:
            kwargs = {}
        return super().__torch_function__(func, types, args, kwargs)

    def __repr__(self):
        return f"LoggingTensor(shape={tuple(self.shape)})"

class MyModel(torch.nn.Module):
    def forward(self, x):
        return x

def my_model_function():
    return MyModel()

def GetInput():
    data = torch.tensor([0.0])
    return data.as_subclass(LoggingTensor)

