import torch
import torch.nn as nn

class _FSDPState(_State):
    def __init__(self) -> None:
        ...
        self.compute_device = torch.device("cuda", torch.cuda.current_device())
        ...

@no_type_check
def _init_param_handles_from_module(
    state: _FSDPState,
    root_module: nn.Module,
    policy: _FSDPPolicy,
    device_id: Optional[Union[int, torch.device]],
    param_init_fn: Optional[Callable[[nn.Module], None]],
    sync_module_states: bool,
) -> _FSDPState:
        ...
        if not hasattr(state, "compute_device"):  # always false and never get compute device from module instance
            state.compute_device = _get_compute_device(
                fully_sharded_module,
                state._ignored_params,
                device_from_device_id,
                state.rank,
            )