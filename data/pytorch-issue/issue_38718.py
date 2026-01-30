import torch

# this function f is exported successfully
@torch.jit.script 
def f(x: Torch.Tensor):
    return torch.nonzero(x)

@torch.jit.script 
def f(x: Torch.Tensor):
    return torch.nonzero(x, as_tuple=True)

@torch.jit._overload_method
def nonzero_(self, input: torch.Tensor, out: Optional[torch.Tensor] = None, as_tuple: bool = False) -> torch.Tensor:
    pass

@torch.jit._overload_method
def nonzero_(self, input: torch.Tensor, out: Optional[torch.Tensor] = None, as_tuple: bool = False) -> List[torch.Tensor]:
    pass

def nonzero_(self, input: torch.Tensor, out: Optional[torch.Tensor] = None, as_tuple: bool = False):
    if not as_tuple:
        if out is not None:
            return torch.nonzero(input, out=out)
        else:
            return torch.nonzero(input)
    else:
        if input.dim() == 0:
            return input.unsqueeze(0).nonzero().unbind(1)
        return input.nonzero().unbind(1)