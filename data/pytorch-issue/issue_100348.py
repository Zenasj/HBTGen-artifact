import torch.nn as nn

import torch
import torch._dynamo


class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        loss = x.mean()
        return {"loss": loss, "loss2": loss.detach()}
        # return {"loss": loss, "loss2": loss.clone().detach()}  # this fixes the issue


mod = Repro()
opt_mod = torch._dynamo.optimize("inductor")(mod)

x = torch.randn((4, 4), requires_grad=True)

vars = mod(x)
print(vars["loss"] is vars["loss2"])
vars["loss"].backward()

vars = opt_mod(x)
print(vars["loss"] is vars["loss2"])
vars["loss"].backward()

def call(args):
    primals_1, = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((), (), device='cuda', dtype=torch.float32)
        buf1 = buf0; del buf0  # reuse
        stream0 = get_cuda_stream(0)
        triton_per_fused_mean_0.run(buf1, primals_1, 1, 16, grid=grid(1), stream=stream0)
        del primals_1
        return (buf1, buf1, )

def compiled_fn(x):
    out = x * 2
    return out, out.view(out.shape)
    
out1, out2 = compiled_fn(x)
out1.t_()  # this should **not** mutate metadata of out2