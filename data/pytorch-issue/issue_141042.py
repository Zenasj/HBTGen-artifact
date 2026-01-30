import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._layer = torch.nn.Linear(2, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self._layer(X)

model = Model()

with torch.inference_mode():
    X = torch.randn(2,)

    # AOTI compile on device="cuda:0"
    device = "cuda:0"
    model = model.to(device=device)
    example_inputs = X.to(device)
    exported_program = torch.export.export(model, (example_inputs,),)
    so_path = torch._inductor.aot_compile(exported_program.module(), (example_inputs,),)

    # inference: OK
    print("-" * 50 + "\n" + f"{device=}")
    aot_model = torch._export.aot_load(so_path, device)
    print(aot_model(X.to(device)))
    print()

    # inference: SEGFAULT
    device = "cuda:1"
    print("-" * 50 + "\n" + f"{device=}")
    aot_model = torch._export.aot_load(so_path, device)
    print(aot_model(X.to(device)))