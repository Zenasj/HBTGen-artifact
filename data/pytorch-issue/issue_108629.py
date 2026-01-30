import torch
import torch.cuda.amp

torch.autocast  # "autocast" is not exported from module "torch"

torch.cuda.amp.GradScaler()  # "GradScaler" is not exported from module "torch.cuda.amp"

torch.cuda.amp.autocast  # "autocast" is not exported from module "torch.cuda.amp"

torch.cuda.amp.custom_fwd  # "custom_fwd" is not exported from module "torch.cuda.amp"

torch.cuda.amp.custom_bwd  # "custom_bwd" is not exported from module "torch.cuda.amp"

import torch

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y

wrapped_add = torch.autocast(device_type="cuda")(add)

reveal_type(wrapped_add)  # Unknown | _Wrapped[..., Unknown, (*args: Unknown, **kwargs: Unknown), Unknown]

reveal_type(arapped_add)  # Type of "wrapped_add" is "(x: Tensor, y: Tensor) -> Tensor"

import torch
import torch.cuda.amp

grad_scaler = torch.cuda.amp.GradScaler()
loss = torch.tensor(1.0)

# Cannot access member "backward" for type "list[Unknown]"
# Cannot access member "backward" for type "tuple[Unknown, ...]"
# Cannot access member "backward" for type "map[Unknown]"
grad_scaler.scale(loss).backward()

reveal_type(grad_scaler.scale(loss))  # Type of "grad_scaler.scale(loss)" is "Tensor"
reveal_type(grad_scaler.scale([loss, loss]))  # Type of "grad_scaler.scale([loss, loss])" is "List[Tensor]"
reveal_type(grad_scaler.scale((loss, loss)))  # Type of "grad_scaler.scale((loss, loss))" is "Tuple[Tensor, ...]"
reveal_type(grad_scaler.scale({loss, loss}))  # Type of "grad_scaler.scale({loss, loss})" is "Iterable[Tensor]"