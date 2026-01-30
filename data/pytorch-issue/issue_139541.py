import torch
from abc import ABC
from dataclasses import dataclass

class Foo(ABC):
    pass

class FooWrapper(Foo):
    def __init__(self, x, y):
        self.x = x
        self.y = y

f = FooWrapper(1, 2)
torch.save(f, "temp.pt")
with torch.serialization.safe_globals([FooWrapper]):
    torch.load("temp.pt")