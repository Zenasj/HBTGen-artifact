import torch


class MyClass(torch.Tensor):
    def foo(self):
        subclasses = MyClass.__subclasses__()
        types_ = tuple(
            torch.Tensor if t in subclasses else t for t in [type(self)]
        )
        return torch.Tensor.__torch_function__(torch.abs, types_, torch.rand(1), {})


def create_subclass(parents):
    class MySubClass(*parents):
        ...

    return MySubClass


def call_foo(x):
    return x.foo()


cls = create_subclass((MyClass,))

call_foo(cls(torch.rand(1000, 1000)))  # works
torch.compile(call_foo, backend='eager')(cls(torch.rand(1000, 1000)))  # fails

import torch


class MyClass(torch.Tensor):
    def foo(self):
        subclasses = MyClass.__subclasses__()
        types_ = tuple(
            torch.Tensor if t in subclasses else t for t in [type(self)]
        )
        return torch.Tensor.__torch_function__(torch.abs, types_, torch.rand(1), {})



def call_foo(x):
    return x.foo()

class MySubClass(MyClass):
    ...

cls = MySubClass

call_foo(cls(torch.rand(1000, 1000)))  # works
torch.compile(call_foo, backend='eager')(cls(torch.rand(1000, 1000)))  # works

import torch


class MyClass(torch.Tensor):
    def foo(self):
        subclasses = [MyClass]
        types_ = tuple(
            torch.Tensor if t in subclasses else t for t in [type(self)]
        )
        return torch.Tensor.__torch_function__(torch.abs, types_, torch.rand(1), {})


def create_subclass(parents):
    class MySubClass(*parents):
        ...

    return MySubClass


def call_foo(x):
    return x.foo()


cls = create_subclass((MyClass,))

call_foo(cls(torch.rand(1000, 1000)))  # works
torch.compile(call_foo, backend='eager')(cls(torch.rand(1000, 1000)))  # works

import torch


class MyClass(torch.Tensor):
    def foo(self):
        subclasses = MyClass.__subclasses__()
        types_ = tuple(
            torch.Tensor for t in [type(self)]
        )
        return torch.Tensor.__torch_function__(torch.abs, types_, torch.rand(1), {})


def create_subclass(parents):
    class MySubClass(*parents):
        ...

    return MySubClass


def call_foo(x):
    return x.foo()


cls = create_subclass((MyClass,))

call_foo(cls(torch.rand(1000, 1000)))  # works
torch.compile(call_foo, backend='eager')(cls(torch.rand(1000, 1000)))  # works

import torch


class MyClass(torch.Tensor):
    def foo(self):
        subclasses = MyClass.__subclasses__()
        types_ = tuple(
            torch.Tensor if t in subclasses else t for t in [type(self)]
        )
        return types_


def create_subclass(parents):
    class MySubClass(*parents):
        ...

    return MySubClass


def call_foo(x):
    return x.foo()


cls = create_subclass((MyClass,))

call_foo(cls(torch.rand(1000, 1000)))  # works
torch.compile(call_foo, backend='eager')(cls(torch.rand(1000, 1000)))  # works

# FAILS THE SAME WAY AS THE CODE ABOVE
import torch


class MyClass(torch.Tensor):
    def foo(self):
        subclasses = MyClass.__subclasses__()
        types_ = tuple(
            torch.Tensor if t in subclasses else t for t in [type(self)]
        )  # this is never used anywhere!
        types_ = (torch.Tensor,)  # because it is overwritten here
        return torch.Tensor.__torch_function__(torch.abs, types_, torch.rand(1), {})


def create_subclass(parents):
    class MySubClass(*parents):
        ...

    return MySubClass


def call_foo(x):
    return x.foo()


cls = create_subclass((MyClass,))

call_foo(cls(torch.rand(1000, 1000)))  # works
torch.compile(call_foo, backend='eager')(cls(torch.rand(1000, 1000)))  # fails

# THIS WORKS!
import torch


class MyClass(torch.Tensor):
    def foo(self):
        subclasses = MyClass.__subclasses__()
        # types_ = tuple(
        #     torch.Tensor if t in subclasses else t for t in [type(self)]
        # )  # commenting out the unused line
        types_ = (torch.Tensor,)
        return torch.Tensor.__torch_function__(torch.abs, types_, torch.rand(1), {})


def create_subclass(parents):
    class MySubClass(*parents):
        ...

    return MySubClass


def call_foo(x):
    return x.foo()


cls = create_subclass((MyClass,))

call_foo(cls(torch.rand(1000, 1000)))  # works
torch.compile(call_foo, backend='eager')(cls(torch.rand(1000, 1000)))  # fails