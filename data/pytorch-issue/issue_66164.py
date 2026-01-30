def foo(x: int = None) -> None:
  ...


# should become

from typing import Optional
def foo(x: Optional[int] = None) -> None:  
    ...