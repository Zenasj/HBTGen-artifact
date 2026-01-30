import torch.nn as nn

import torch


encoder_layer = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.rand(10, 32, 512)
out = transformer_encoder(src)

torch.onnx.export(encoder_layer, src, "_transformer_encoder.onnx", opset_version=17)