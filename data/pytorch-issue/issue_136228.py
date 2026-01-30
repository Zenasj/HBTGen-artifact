import torch

state_dict_options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
torch.distributed.checkpoint.state_dict.set_model_state_dict(model, local_state_dict, options=state_dict_options)

local_state_dict[key] = DTensor.from_local(
    local_tensor, local_state.device_mesh, local_state.placements,
)

local_state_dict[key] = DTensor.from_local(
    local_tensor, local_state.device_mesh, local_state.placements, shape=full_tensor.shape, stride=full_tensor.stride()
)