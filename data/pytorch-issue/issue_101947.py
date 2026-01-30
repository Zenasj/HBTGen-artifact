import torch.nn as nn

import torch
import torch.nn.functional as F
from typing import Any, Sequence, Union, Optional, Mapping
import numpy as np
import io
import onnxruntime
from torch.types import Number
_InputArgsType = Optional[
    Union[torch.Tensor, int, float, bool, Sequence[Any], Mapping[str, Any]]
]
_NumericType = Union[Number, torch.Tensor, np.ndarray]
_OutputsType = Sequence[_NumericType]
def run_ort(
    onnx_model: Union[str, torch.onnx.ExportOutput],
    pytorch_inputs: Sequence[_InputArgsType],
) -> _OutputsType:
    """Run ORT on the given ONNX model and inputs

    Used in test_fx_to_onnx_with_onnxruntime.py

    Args:
        onnx_model (Union[str, torch.onnx.ExportOutput]): Converter ONNX model
        pytorch_inputs (Sequence[_InputArgsType]): The given torch inputs

    Raises:
        AssertionError: ONNX and PyTorch should have the same input sizes

    Returns:
        _OutputsType: ONNX model predictions
    """
    if isinstance(onnx_model, torch.onnx.ExportOutput):
        buffer = io.BytesIO()
        onnx_model.save(buffer)
        ort_model = buffer.getvalue()
    else:
        ort_model = onnx_model
    print("before")
    session = onnxruntime.InferenceSession(
        ort_model, providers=["CPUExecutionProvider"]
    )
    print("after")
    input_names = [ort_input.name for ort_input in session.get_inputs()]

    if len(input_names) != len(pytorch_inputs):
        raise AssertionError(
            f"Expected {len(input_names)} inputs, got {len(pytorch_inputs)}"
        )

    return session.run(
        None, {k: v.cpu().numpy() for k, v in zip(input_names, pytorch_inputs)}
    )


class SingleOpModel(torch.nn.Module):
    """Test model to wrap around a single op for export."""

    def __init__(self, op, kwargs):
        super().__init__()
        self.operator = op
        self.kwargs = kwargs

    def forward(self, *args):
        return self.operator(*args, **self.kwargs)


model = SingleOpModel(F.elu, {})
x = torch.rand((10), dtype=torch.float16)
export_output = torch.onnx.dynamo_export(
    model,
    *(x, 1.0),
    {},
    export_options=torch.onnx.ExportOptions(
        opset_version=18,
        op_level_debug=False,
        dynamic_shapes=False,
    ),
)

onnx_format_args = export_output.adapt_torch_inputs_to_onnx(*(x, 1.0), {})
ort_outputs = run_ort(export_output, onnx_format_args)