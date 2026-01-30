from dataclasses import dataclass
from typing import TypeVar


class Sequential(list):
    def __add__(self, other):
        if isinstance(other, __class__):
            return __class__([*self, *other])
        else:
            raise TypeError(f"unsupported operand type(s) for +: {type(self)} and {type(other)}")
        

@dataclass
class Module(object):
    name: str = ""
    
    
if __name__ == '__main__':
    a = Module(name="a")
    b = Module(name="b")
    c = Sequential([a, b])
    
    d = Module(name="d")
    e = Module(name="e")
    g = Sequential([d, e])
    
    f = c + g
    print(f)