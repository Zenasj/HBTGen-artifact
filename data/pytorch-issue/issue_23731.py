import torch

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

class MyOptimizer(Optimizer):
    def __init__(self, params: List[torch.Tensor]):
        defaults = dict(lr=0.01)
        super(MyOptimizer, self).__init__(params, defaults)

    def f(self):
        print(self.state)
        print(self.defaults)
        print(self.param_groups)