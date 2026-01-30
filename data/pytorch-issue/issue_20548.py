from typing import Iterable, Union, Callable, Optional
from .. import Tensor

_params_t = Union[Iterable[Tensor], dict]

class Optimizer:
    def __init__(self, params: _params_t) -> None: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state_dict: dict) -> None: ...
    def zero_grad(self) -> None: ...
    def step(self, closure: Optional[Callable[[], float]]=...) -> None: ...
    def add_param_group(self, param_group: dict) -> None: ...

_params_t = Union[Iterable[torch.Tensor], Iterable[Dict]]

from typing import Union, Iterable, Dict
import torch
from torch.optim import Adam, Optimizer

def _default_optimizer(params: Union[Iterable[torch.Tensor], Iterable[Dict]]) -> Optimizer:
    return Adam(params)