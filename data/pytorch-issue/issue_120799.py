import torch

class A(torch.Tensor):
    pass

class B(torch.Tensor):
    def __torch_function__(self, func, types, args, kwargs=None):
        return super().__torch_function__(func, types, args, kwargs)

class C(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args, kwargs=None):
        return super().__torch_function__(func, types, args, kwargs)


@torch.compile(backend="eager")
def fn(cls):
    return cls() + 1


fn(A)
print("A is good")

try:
    fn(B)
except Exception as e:
    assert "UserDefinedObjectVariable(B) is not a constant" in str(e)
    print("B failed")

fn(C)
print("C is good")

import torch

class A(torch.Tensor):
    @classmethod
    def foo(cls):
        return cls

class D(A):
    def foo(self):
        return super().foo()

class E(A):
    @classmethod
    def foo(self):
        return super().foo()

@torch.compile(backend="eager")
def run():
    v1 = A().foo()
    v2 = D().foo()
    v3 = E().foo()
    return v1, v2, v3
print(run())

@torch.compile(backend="eager")
def run():
    v1 = A().foo()
    v2 = D().foo()
    return v1, v2
print(run())

@torch.compile(backend="eager")
def run():
    v1 = A().foo()
    return v1
print(run())

@torch.compile(backend="eager")
def run():
    v1 = A().foo()
    v2 = A().foo()
    return v1, v2
print(run())