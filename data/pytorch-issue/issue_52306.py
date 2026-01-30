import torch

@torch.jit.script
class MyClass(object):
   def __init__(self, x: int):
      self.x = x

   def set(self, val: int):
       self.x: float = val

def fn(a: MyClass, b: int):
    a.set(b)
    return a.x

y = MyClass(1)
print("Eager: ", fn(y, "a string"))
print(type(y.x))