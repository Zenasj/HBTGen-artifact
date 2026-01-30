import torch
from xformers.checkpoint import selective_checkpoint_wrapper

def my_fn(t, w):
    b, _ = t.shape
    idxs = (torch.randperm(b, device=t.device))[:b // 2]
    sub_in = t[idxs]
    sub_out = torch.matmul(sub_in, w)
    out = torch.index_add(t, dim=0, source=sub_out, index=idxs, alpha=0.5)
    return out

fn = torch.compile(
    selective_checkpoint_wrapper(
        my_fn,
        policy_fn=lambda mode, func, *args, **kwargs: "randperm" in str(func)
    ),
    options={"triton.cudagraphs": True}
)

t = torch.randn((512, 2048), dtype=torch.bfloat16, device="cuda", requires_grad=True)
w = torch.randn((2048, 2048), dtype=torch.bfloat16, device="cuda", requires_grad=True)
fn(t, w.t())