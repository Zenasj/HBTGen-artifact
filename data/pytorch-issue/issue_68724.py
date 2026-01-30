import torch.nn as nn

import numpy as np
import torch
from torch import nn
import onnx
import onnxruntime


dummy_in = torch.ones(1, 3, 100, 100)


class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(16)

    def forward(self, x: torch.Tensor):
        return self.norm(self.conv(x))


class TraceInScriptWithIf(nn.Module):
    def __init__(self):
        super().__init__()
        self.traced_conv_one_block = torch.jit.trace(  # type:ignore
            ConvBlock().eval(), (dummy_in))
        print(self.traced_conv_one_block.graph)
        self.traced_conv_two = torch.jit.trace(  # type:ignore
            ConvBlock().eval(), (dummy_in))
        print(self.traced_conv_two.graph)

    def forward(self, x):
        if x.sum() > 0:
            return self.traced_conv_one_block(x)
        return self.traced_conv_two(x)


scripted_control_flow = torch.jit.script(TraceInScriptWithIf())  # type:ignore
print(scripted_control_flow.graph)
dummy_out = scripted_control_flow(dummy_in)
# Convert onnx
onnx_model_name = 'script_in_trace_with_if.onnx'
torch.onnx.export(
    scripted_control_flow,
    (dummy_in),
    onnx_model_name,
    do_constant_folding=True,
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "n", 2: "h", 3: "w"},
        "output": {0: "n", 2: "h", 3: "w"},
    },
)
# Print onnx graph
onnx_model = onnx.load(onnx_model_name)
print(onnx.helper.printable_graph(onnx_model.graph))  # type: ignore

# Test Onnx Graph


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


def all_close(tensor: torch.Tensor, ndarray: np.ndarray) -> str:
    return 'OK' if np.allclose(to_numpy(tensor), ndarray, rtol=1e-03, atol=1e-05) else 'not OK'


ort_session = onnxruntime.InferenceSession(
    onnx_model_name, providers=["CUDAExecutionProvider"],
)

ort_dummy_in_dict = {"input": to_numpy(dummy_in)}
ort_dummy_out_list = ort_session.run(None, ort_dummy_in_dict)
ort_dummy_out = ort_dummy_out_list[0]
print('Check ort inference')
print('ort_zeros_out', all_close(dummy_out, ort_dummy_out))

# Add a zero between Conv and BatchNorm
class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(16)

    def forward(self, x: torch.Tensor):
        return self.norm(self.conv(x) + 0) # stop fusing Conv and BatchNorm