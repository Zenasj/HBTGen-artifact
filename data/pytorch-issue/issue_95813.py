import torch

if param.device == cpu_device:
            # NOTE: This includes moving ignored modules' parameters.
            module = module.to(device_from_device_id)
            # TODO: This is a temporary fix to move already-constructed
            # `FlatParameter`s back to CPU if needed. This is needed to
            # make CPU offload work with `device_id`.
            for submodule in module.modules():
                if (
                    isinstance(submodule, fsdp_file.FullyShardedDataParallel)
                    and submodule.cpu_offload.offload_params
                ):
                    for handle in submodule._handles:
                        handle.flat_param_to(torch.device("cpu"))

module = module.to(device_from_device_id)

_move_modules_not_under_fsdp_offload_to_device(module, device_from_device_id)

def _move_modules_not_under_fsdp_offload_to_device(module, device):    
    if not (
        isinstance(module, fsdp_file.FullyShardedDataParallel)
        and module.cpu_offload.offload_params
    ):
        for key, param in module._parameters.items():
            if param is not None:
                module._parameters[key] = param.to(device, non_blocking=True)

        for submodule in module.children():
            _move_modules_not_under_fsdp_offload_to_device(submodule, device)