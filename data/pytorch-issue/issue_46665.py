import torch
import torch.nn as nn
import contextlib
import mock

# toy contextmanager decorator
@contextlib.contextmanager
def do_nothing():
    yield

# real use case (same error to above when symbolically tracing)
def onnx_compatible_interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    # NOTE: The input dimensions are interpreted in the form:
    # `mini-batch x channels x [optional depth] x [optional height] x width`.
    if size is None and scale_factor is not None:
        if mode == "nearest" and input.dim() == 4:
            if isinstance(scale_factor, (int, float)):
                height_scale, width_scale = (scale_factor, scale_factor)
            else:
                assert isinstance(scale_factor, tuple) or isinstance(scale_factor, list)
                assert len(scale_factor) == 2
                height_scale, width_scale = scale_factor
            return torch.ops._caffe2.ResizeNearest(
                input, order="NCHW", width_scale=width_scale, height_scale=height_scale
            )

    return nn.functional.interpolate(input, size, scale_factor, mode, align_corners)

@contextlib.contextmanager
def mock_torch_nn_functional_interpolate():
    if torch.onnx.is_in_onnx_export():
        with mock.patch(
            "torch.nn.functional.interpolate", side_effect=onnx_compatible_interpolate
        ):
            yield
    else:
        yield

class M(nn.Module):
    def __init__(self):
        super().__init__()
    
    # real use case
    # @mock_torch_nn_functional_interpolate()
    # toy repro
    @do_nothing()
    def forward(self, x):
        return x
    
m = M()
gm = torch.fx.symbolic_trace(m)