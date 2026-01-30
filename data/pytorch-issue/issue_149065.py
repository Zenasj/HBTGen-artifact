import numpy as np
import random

from torch.export import Dim
from pathlib import Path

import onnx
import onnxruntime
import torch

model = model
model.load_state_dict(checkpoint.get("state_dict"), strict=True)
model.eval()

with torch.no_grad():
    data = torch.randn(1, 3, 256, 256)
    torch_outputs = model(data)

    example_inputs = (data.cuda(),)

    batch_dim = Dim("batch_size", min=1, max=16)
    onnx_program = torch.onnx.export(
        model=model.cuda(),
        args=example_inputs,
        dynamo=True,
        input_names=["images"],
        output_names=["logits"],
        opset_version=20,
        dynamic_shapes=({0: batch_dim},),
    )
    onnx_program.optimize()
    onnx_program.save(str(ONNX_MODEL))
    del onnx_program
    del model

onnx_model = onnx.load(str(ONNX_MODEL))
onnx.checker.check_model(onnx_model)
num_nodes = len(onnx_model.graph.node)
print(f"Number of nodes in the ONNX model: {num_nodes}")

# Inspect inputs
print("Model Inputs:")
for inp in onnx_model.graph.input:
    dims = [dim.dim_value if dim.HasField("dim_value") else dim.dim_param for dim in inp.type.tensor_type.shape.dim]
    print(f"{inp.name}: {dims}")

# Inspect outputs
print("\nModel Outputs:")
for out in onnx_model.graph.output:
    dims = [dim.dim_value if dim.HasField("dim_value") else dim.dim_param for dim in out.type.tensor_type.shape.dim]
    print(f"{out.name}: {dims}")

del onnx_model

onnx_inputs = [tensor.numpy(force=True) for tensor in example_inputs]
ort_session = onnxruntime.InferenceSession(str(ONNX_MODEL), providers=["CPUExecutionProvider"])

onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)}

# ONNX Runtime returns a list of outputs
onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]

assert len(torch_outputs) == len(onnxruntime_outputs)
for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
    torch.testing.assert_close(torch_output.cpu(), torch.tensor(onnxruntime_output))

print("All tests passed")

from pathlib import Path

import numpy
import onnx
import onnxruntime

ROOT = Path(__file__).resolve().parent.parent
ONNX_MODEL = ROOT / "model.onnx"

onnx_model = onnx.load(str(ONNX_MODEL))

onnx_inputs = [numpy.random.randn(4, 3, 256, 256).astype(numpy.float32)]
ort_session = onnxruntime.InferenceSession(str(ONNX_MODEL), providers=["CPUExecutionProvider"])

onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)}

# ONNX Runtime returns a list of outputs
onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]