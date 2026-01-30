import torch.nn as nn

import io
import numpy as np
import onnxruntime as ort
import torch

from torch import nn

num_features = 32
batch = 4
img_dims = 1200
dims = (2, 3)
model = nn.InstanceNorm2d(num_features=num_features, eps=0., affine=False)

torch.manual_seed(42)
x = 1e-3 * torch.randn(batch, num_features, img_dims, img_dims)
y = model(x)

bitstream = io.BytesIO()
torch.onnx.export(
    model=model,
    args=(x,),
    f=bitstream,
    input_names=["x"],
    opset_version=11,
)
bitstream_data = bitstream.getvalue()
ort_session = ort.InferenceSession(bitstream_data)
ort_inputs = {"x": x.numpy()}
ort_outputs = ort_session.run(None, ort_inputs)
ort_y = ort_outputs[0]

np.testing.assert_allclose(ort_y, y.detach().numpy(), rtol=1e-5, atol=1e-5)

x = 1e-3 * torch.rand(batch, num_features, img_dims, img_dims)