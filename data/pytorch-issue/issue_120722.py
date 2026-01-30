import torch
import torch.nn as nn

use_full_state_dict = not state.is_model_ddp and isinstance(state_dict_model, FSDP)
state_dict_context = fsdp_state_dict_type_context(
  original_model, state_dict_type='full') if use_full_state_dict else contextlib.nullcontext()

with state_dict_context:
  state_dict = state_dict_model.state_dict()

from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

state_dict = get_model_state_dict(
  state_dict_model,
  options=StateDictOptions(
    full_state_dict=True,
    cpu_offload=True,
  ),
)

cpu_offload = True
def dtensor_to_tensor_hook(
    module: nn.Module,
    state_dict: Dict[str, Any],
    prefix: str,
    *args: Any,
) -> Dict[str, Any]:
    dtensor_fqns = []
    for fqn in state_dict.keys():
        tensor = state_dict[fqn]
        if isinstance(tensor, DTensor):
            dtensor_fqns.append(fqn)
            tensor = tensor.full_tensor()
            if dist.get_global_rank() == 0:
                if cpu_offload:
                    tensor = tensor.cpu()
                state_dict[fqn] = tensor
    if dist.get_global_rank() != 0:
        for fqn in dtensor_fqns:
            del state_dict[fqn]
    return state_dict

hooks = []
for name, module in state_dict_model.named_modules():
    if isinstance(module, FSDP):
        hooks.append(
            module._register_state_dict_hook(
                dtensor_to_tensor_hook))