py
import os

import torch
import torch.distributed._functional_collectives as funcol

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "2743"
torch.distributed.init_process_group(backend="nccl")


def scale(t):
    scale = torch.finfo(torch.float8_e4m3fn).max / t.abs().amax(dim=-1, keepdim=True).float()
    t = t.mul(scale).to(torch.float8_e4m3fn)
    return t, scale


def fp8_rowwise_backward(in_, w, out_grad):
    out_grad_fp8, scale_out_grad = scale(out_grad)
    w_fp8, scale_w = scale(w.t().contiguous())
    out_grad_fp8 = funcol.all_gather_tensor(
        out_grad_fp8, gather_dim=0, group=torch.distributed.group.WORLD
    )
    scale_out_grad = funcol.all_gather_tensor(
        scale_out_grad, gather_dim=0, group=torch.distributed.group.WORLD
    )
    in_grad = torch._scaled_mm(
        out_grad_fp8, w_fp8.t(), scale_a=scale_out_grad, scale_b=scale_w.t(), out_dtype=torch.bfloat16
    )

    out_grad = funcol.all_gather_tensor(
        out_grad.t().contiguous(), gather_dim=0, group=torch.distributed.group.WORLD
    )
    w_grad = out_grad @ in_

    return in_grad, w_grad


in_ = torch.randn((3072, 4096), device="cuda", dtype=torch.bfloat16)
w = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
out_grad = torch.randn((3072, 4096), device="cuda", dtype=torch.bfloat16)

eager_in_grad, eager_w_grad = fp8_rowwise_backward(in_, w, out_grad)
compile_in_grad, compile_w_grad = torch.compile(fp8_rowwise_backward)(in_, w, out_grad)

assert torch.testing.assert_close(compile_w_grad, eager_w_grad)