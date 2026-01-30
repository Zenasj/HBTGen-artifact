import torch

# 1 add device parameter
def all_gather_object(object_list, obj, group=None, device=None):
    current_device = device or _get_pg_device(group)

# 2 Add judgment for custom backend
def barrier(group=GroupMember.WORLD, async_op=False, device_ids=None):
    if device_ids is not None:
        if (get_backend(group) != Backend.NCCL) or (get_backend(group) in Backend._plugins) :
            raise RuntimeError(
                "Function argument device_ids not supported "
                "for the selected backend {}".format(get_backend(group))
            )

# 3 add judgment for custom device
if not torch.cuda.is_available() or not custom_device_is_available:
    raise ValueError("Subgroups can only be created when CUDA or custom device is available")