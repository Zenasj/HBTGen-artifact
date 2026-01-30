import torch.nn as nn

import torch
from torch import Tensor
import io
from typing import Optional
import onnx
import onnxruntime as ort

class LoopNoneInput(torch.nn.Module):
    def forward(self, x):
        y: Optional[Tensor] = None
        for _ in range(x.size(0)):
            y = x
        return y


f = io.BytesIO()
x = torch.ones(1)
dynamic_axis_name = "condition"
torch.onnx.export(
    torch.jit.script(LoopNoneInput()),
    (x,),
    f,
    opset_version=16,
    # Ensure condition is not constant
    dynamic_axes={"x": {0: dynamic_axis_name}},
    input_names=["x"],
)

model = onnx.load_model_from_string(f.getvalue())
onnx.checker.check_model(model)

ort_input = {
    "x":x.cpu().numpy()
}
sess = ort.InferenceSession(model.SerializeToString() , ort.SessionOptions(), providers=['CPUExecutionProvider'])
out = sess.run(None, ort_inputs)
# Alternatively, one can use the following line to replace running ORT
# same outcome can be obtained.
#onnx.shape_inference.infer_shapes(model, strict_mode=True)