import torch

@torch.jit.script
def upd() -> Dict[str, int]:
    a: Dict[str, int] = {}
    for i in range(3):
        a.update({'a': i})
    return a
        
print(upd())
# > {'a': 0}