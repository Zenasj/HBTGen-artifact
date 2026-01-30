import torch
import torch.nn as nn

class M(torch.nn.Module):
    def forward(self, dilate_flag):
        return dilate_flag.item()

input1 = (torch.tensor([1], dtype=torch.bool, device="cuda"),)
model = M().cuda()

ep = torch.export.export(model, input1, strict=True)
path = torch._inductor.aot_compile(ep.module(), input1)
aot_model = torch._export.aot_load(path, device="cuda")
actual_output = aot_model(*input1)

class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dilate_flag = 0

    def forward(self, dilate_flag):
        self.dilate_flag = dilate_flag.item()

        def true_fn(dilate_flag):
            return dilate_flag.clone()

        def false_fn(dilate_flag):
            return dilate_flag.clone()

        torch.cond(
            self.dilate_flag,
            true_fn,
            false_fn,
            (dilate_flag,),
        )
        return self.dilate_flag

input1 = (torch.tensor([1], dtype=torch.bool, device="cuda"),)
input2 = (torch.tensor([0], dtype=torch.bool, device="cuda"),)
inputs = (input1, input2)
model = M().cuda()

for input in inputs:
    expected_output = model(*input)

    ep = torch.export.export(model, input, strict=False)
    path = torch._inductor.aot_compile(ep.module(), input)
    aot_model = torch._export.aot_load(path, device="cuda")
    actual_output = aot_model(*input)

    assert (
        expected_output == actual_output
    ), f"henry they are not equal {expected_output} != {actual_output}"