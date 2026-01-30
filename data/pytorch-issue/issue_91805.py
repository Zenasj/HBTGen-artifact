from enum import Enum
import inspect

class Color(Enum):
    RED = 1
    GREEN = 2

def print_routines(cls):
    print(cls.__name__)
    for name in cls.__dict__:
        fn = getattr(cls, name)
        if inspect.isroutine(fn):
            print(name, fn, f"has_globals: {hasattr(fn, '__globals__')}")

print_routines(Color)