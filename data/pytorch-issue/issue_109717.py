import sys
import inspect
from enum import Enum

class IntColor(int, Enum):
    RED = 1
    GREEN = 2

class Color(Enum):
    RED = 1
    GREEN = 2

def get_methods(cls):
    def predicate(m):
        if not inspect.isfunction(m) and not inspect.ismethod(m):
            return False
        return m.__name__ in cls.__dict__
    return inspect.getmembers(cls, predicate=predicate)

if __name__ == "__main__":
    print(sys.version)
    print(f"IntColor methods {get_methods(IntColor)}")
    print(f"Color methods {get_methods(Color)}")