import torch
import torch.nn as nn

class M(torch.nn.Module):
    def forward(self, x, flag):
        # without the following line, it can run fine
        flag = flag.item()

        def true_fn(x):
            return x.clone()

        return torch.cond(flag > 0, true_fn, true_fn, [x])

input = (
    torch.rand(28, 28, device="cuda"),
    torch.tensor(1),
)
model = M().cuda()

_ = model(*input)

ep = torch.export.export(model, input, strict=False)
path = torch._inductor.aot_compile(ep.module(), input)
aot_model = torch._export.aot_load(path, device="cuda")
torch.testing.assert_close(aot_model(*input), model(*input))