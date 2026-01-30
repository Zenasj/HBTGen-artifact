import torch.nn as nn

import torch
import torch._dynamo as torchdynamo

class Foo(torch.nn.Module):
    def __init__(
        self,
        input_dim,
    ):
        super().__init__()
        self.torch_module = torch.nn.LayerNorm(
            input_dim, eps=1e-5, elementwise_affine=True
        )

    def forward(self, input):
        output = torch.nn.functional.layer_norm(
            input,
            self.torch_module.normalized_shape,
            self.torch_module.weight,
            self.torch_module.bias,
            self.torch_module.eps,
        ).type_as(input)
        return output

mod = Foo(128)
inp = torch.randn(3, 128)
gm, _ = torchdynamo.export(mod, inp, aten_graph=True, tracing_mode="symbolic")
print(gm.graph)