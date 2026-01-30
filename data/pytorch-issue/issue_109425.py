import torch.nn as nn

import io
import torch
from torch.ao.quantization import (
    get_default_qconfig_mapping,
    get_default_qconfig,
)
from torch.ao.quantization.quantize_fx import (
    prepare_fx,
    convert_fx,
)


class TransposeModel(torch.nn.Module):
    def __init__(self, dims: torch.types._size):
        super().__init__()

        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = torch.permute(x, self.dims)
        x = x.permute(self.dims)

        return x


if __name__ == "__main__":
    model = TransposeModel(dims=(0, 2, 1))
    qconfig = get_default_qconfig("qnnpack")
    qconfig_mapping = get_default_qconfig_mapping("qnnpack") \
        .set_global(qconfig)

    inputs = {
        "x": torch.reshape(torch.arange(0, 99, dtype=torch.float32), (1, 3, 33)),
    }

    prepared = prepare_fx(
        model=model,
        qconfig_mapping=qconfig_mapping,
        example_inputs=tuple((v for v in inputs.values())),
    )

    quantized = convert_fx(
        graph_module=prepared,
        qconfig_mapping=qconfig_mapping,
    )

    quantized.graph.print_tabular()

    script_module = torch.jit.script(
        obj=quantized,
        example_inputs=[tuple((v for v in inputs.values()))],
    )

    with io.BytesIO() as f:
        torch.onnx.export(
            f=f,
            model=script_module,
            args=[v for v in inputs.values()],
            input_names=["x"],
            output_names=["x"],
            verbose=True,
        )