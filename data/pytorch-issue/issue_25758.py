import torch
from typing import Dict, Tuple, List

@torch.jit.script
def foo(v: Dict[int, int], k: int):
    if k in v:
        return True
    else:
        return False

# @torch.jit.script
# def foo1(v:List[int], k: int):
#     if k in v:
#         return True
#     else:
#         return False

# @torch.jit.script
# def foo2(v:Tuple[int], k: int):
#     if k in v:
#         return True
#     else:
#         return False


def python(v, k):
    if k in v:
        return True
    else:
        return False

if __name__ == "__main__":
    a = [1,2,3,4,5,6]
    b = 3
    print(python(a,b))