import torch.nn as nn

from io import BytesIO

import onnxruntime
import pytest
import torch


class MyThing(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # NOTE will work if these are not tensors but just floats
        self.min_val = torch.tensor([47.0])
        self.max_val = torch.tensor([53.0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(self.min_val, self.max_val)
        return x


@pytest.mark.xfail(reason="This test is expected to fail")
def test_broken_capping() -> None:
    module = MyThing()
    x = torch.tensor([torch.nan])
    y_torch = module.forward(x)
    model_bytes = BytesIO()
    torch.onnx.export(module, x, model_bytes, input_names=["input"])
    model_bytes.seek(0)

    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess = onnxruntime.InferenceSession(model_bytes.read(), sess_options=sess_options)
    y_onnx = sess.run(None, {"input": x.numpy()})[0]
    
    # Will fail as y_onnx = [47.0] and y_torch = [nan]
    assert torch.allclose(y_torch, torch.tensor(y_onnx), equal_nan=True)