import torch.nn as nn

from typing import Tuple

import numpy as np
import onnxruntime as ort
import torch
from numpy.testing import assert_almost_equal
from torch import Tensor

import onnx


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


n_classes = 5

static_weights = [
    [0, 0.8, 0.7, 0.6, 0.4],
    [0.8, 0, 0, 0, 0.6],
    [0.8, 0, 0, 0.4, 0],
    [0.8, 0, 0.7, 0, 0.4],
    [0.8, 0.7, 0, 0.6, 0],
]

# Define test input data.
in_values = torch.tensor([[0.67, 0.61, 0.46, 0.53, 0.41]], dtype=torch.float)
topk = torch.tensor(4, dtype=torch.long)

# Define expected output data.
out_values = torch.tensor([1.608, 0.846, 0.84, 0.832], dtype=torch.float)
out_indices = torch.tensor([0, 4, 2, 3], dtype=torch.long)


class TestFCNModel(torch.nn.Module):
    def __init__(self):
        super(TestFCNModel, self).__init__()

        # Set FCN and load the pre-computed weights.
        self.fcn = torch.nn.Linear(n_classes, n_classes)
        self.fcn.bias.data = torch.zeros(n_classes)
        self.fcn.weight.data.copy_(
            np.transpose(torch.from_numpy(np.array(static_weights)))
        )

    def forward(
        self,
        in_values: Tensor,
        topk: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        scores = scores = torch.sum(self.fcn(in_values), dim=0)
        return torch.topk(scores, topk)


# Test PyTorch model. # Works.
raw_model = TestFCNModel()
raw_val, raw_idx = raw_model(in_values, topk)
assert_almost_equal(to_numpy(raw_val), out_values, decimal=4)
assert_almost_equal(to_numpy(raw_idx), out_indices)

# Convert to TorchScript, then to ONNX.
torchscript_model = torch.jit.script(TestFCNModel())
model_path = "./fcn_test.onnx"
torch.onnx.export(
    model=torchscript_model,  # model being run
    args=(
        in_values,
        topk,
    ),  # model input (or a tuple for multiple inputs)
    f=model_path,  # where to save the model (can be a file or file-like object)
    verbose=True,
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=11,  # the ONNX version to export the model to
    input_names=[
        "input",
        "topk",
    ],  # the model's input names
    output_names=["values", "indices"],  # the model's output names
    example_outputs=[(out_values, out_indices)],
)
# ONNX checker, ok.
onnx.checker.check_model(onnx.load(model_path))

# ONNX init session, !!!  NOT WORKING  !!!
ort_session = ort.InferenceSession(model_path)
ort_inputs = {
    "input": to_numpy(in_values),
    "topk": to_numpy(topk),
}
onnx_val, onnx_idx = ort_session.run(None, ort_inputs)
assert_almost_equal(onnx_val, out_values, decimal=4)
assert_almost_equal(onnx_idx, out_indices)

class TestMatmulModel(torch.nn.Module):
    def __init__(self):
        super(TestMatmulModel, self).__init__()
        # Load weights as variable
        self.weights = torch.from_numpy(np.array(static_weights).astype(np.float32))

    def forward(
        self,
        in_values: Tensor,
        topk: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        # Per-item's final score.
        scores = torch.sum(torch.matmul(in_values, self.weights), dim=0)
        return torch.topk(scores, topk)