import torch

def f() -> int:  # Mypy says: `error: Missing return statement`
    with torch.no_grad():
        return 1

from typing import Generator
from contextlib import contextmanager

@contextmanager
def swallow_zerodiv() -> Generator[None, None, None]:
    try:
        yield None
    except ZeroDivisionError:
        pass
    finally:
        pass

def div(a: int, b: int) -> float:  # This function seems `(int, int) -> float` but actually `(int, int) -> Optional[float]` because ` return a / b` may be swallowed
    with swallow_zerodiv():
        return a / b

if __name__ == '__main__':
    result = div(1, 0)
    print(result, type(result))  # None <class 'NoneType'>