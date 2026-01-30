import torch

class MyClass:
    foo: int = 1

@torch.compile(fullgraph=True)
def func(x, m):
    # return x + getattr(MyClass, "foo", 0)
    if getattr(type(m), "foo", 0):
        return x + MyClass.foo
    return x

m = MyClass()
func(torch.zeros(()), m)
func(torch.zeros(()), 0)

def call_hasattr(self, tx, obj, attr):
        if attr.is_python_constant():
            name = attr.as_python_constant()
            if isinstance(obj, BuiltinVariable):
                return super(type(obj), obj).call_hasattr(tx, attr)
            return obj.call_hasattr(tx, name)