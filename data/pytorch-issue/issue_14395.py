import torch
import torch.nn as nn

avg_pool = nn.AdaptiveAvgPool3d(1)

torch_out = torch.onnx._export(model,
                                   input,
                                   "model_caff2.onnx",
                                   export_params=False)

import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend
model = onnx.load(onnx_file_path)
prepared_backend = onnx_caffe2_backend.prepare(model)
W = {model.graph.input[0].name: input_tensor.data.numpy()}
c2_out = prepared_backend.run(W)[0]