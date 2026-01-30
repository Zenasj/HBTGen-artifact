import torch

def f(x):
    return (x + x).to(torch.int16)

x = torch.tensor(128, dtype=torch.uint8)
print(f(x))
print(torch.compile(f)(x))

tensor(0, dtype=torch.int16)
tensor(256, dtype=torch.int16)

@staticmethod
def add(a, b):
   return f"decltype({a})({a} + {b})"

@staticmethod
def add(a, b):
   return f"decltype({a})({a} + {b})"