@torch.script.jit
def gradient(y, x):
    # grad_outputs = [torch.ones_like(y)]
    grad_outputs = torch.jit.annotate(Optional[Tensor], torch.ones_like(y))
    # grad_outputs = torch.jit.annotate(Optional[List[Optional[Tensor]]], [torch.ones_like(y)]) -> "Expected a List type hint but instead got Optional[List[Optional[Tensor]]]"
    grad = torch.autograd.grad(
        [y], [x], [grad_outputs], create_graph=True, retain_graph=True
    )[0]
    return grad

import torch
from typing import List, Optional

# First version with return value potentially being None
@torch.jit.script
def gradient(y: torch.Tensor, x: torch.Tensor) -> Optional[torch.Tensor]:
    grad_outputs : List[Optional[torch.Tensor]] = [ torch.ones_like(y) ]
    grad = torch.autograd.grad([y,], [x], grad_outputs=grad_outputs, create_graph=True)
    
    # optional type refinement using an if statement
    if grad is not None:
        grad =grad[0]
        
    return grad # grad can be None here, so it is Optional[torch.Tensor]

# Second version with return value being 0 gradient if actually None
@torch.jit.script
def gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    grad_outputs : List[Optional[torch.Tensor]] = [ torch.ones_like(y) ]
    grad = torch.autograd.grad([y,], [x], grad_outputs=grad_outputs, create_graph=True)
    
    # optional type refinement using an if statement
    if grad is None:
        grad = torch.zeros_like(x)
    else:
        grad = grad[0]
     
    # optional type refinement using an assert
    assert grad is not None
    return grad # Now grad is always a torch.Tensor instead of Optional[torch.Tensor]