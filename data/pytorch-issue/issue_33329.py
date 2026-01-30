import torch

@torch.jit.script
def send_rpc_async(dst_worker_name, user_func_qual_name, tensor):
    # type: (str, str, Tensor) -> None
    rpc._rpc_async_torchscript(
        dst_worker_name, user_func_qual_name, args=(tensor,)
    )