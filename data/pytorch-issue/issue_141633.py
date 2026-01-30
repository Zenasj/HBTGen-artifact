import torch
import torch.nn as nn

class FakeParam(object):
    def __init__(
        self,
        data: torch.Tensor,
    ):
        self.data = data


class FakeLinear(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
    ):
        super().__init__()

        self.param = FakeParam(
            torch.empty(
                output_size,
                input_size,
                device=torch.device("cuda"),
                dtype=torch.float32,
            ),
        )

    def forward(self, input_):
        return torch.matmul(input_, self.param.data)


model = FakeLinear(512, 512).cuda()
input = (torch.rand(512, 512, dtype=torch.float32, device="cuda"),)

# sanity check
_ = model(*input)
# sanity check 2
_ = torch.compile(model, fullgraph=True)(*input)

ep = torch.export.export(model, input, strict=False)
path = torch._inductor.aot_compile(ep.module(), input)
aot_model = torch._export.aot_load(path, device="cuda")
aot_model(*input)

print("done")