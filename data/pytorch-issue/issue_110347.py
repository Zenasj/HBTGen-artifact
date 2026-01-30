import torch.nn as nn

from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(15, 96)
        enc_layer = nn.TransformerEncoderLayer (
            96,
            nhead=12,
            dim_feedforward=96,
            dropout=0.2,
            batch_first=True
        )
        self.attn_layers = nn.TransformerEncoder(
            enc_layer,
            num_layers=10,
            enable_nested_tensor=True
        )

    def forward(self, x):
        x = self.emb(x)
        return self.attn_layers(x)

import torch

model = Model()
torch.onnx.export(
    model,
    args=(torch.randint(0, 15, (1, 20))),
    f="model.onnx",
    opset_version=16,
    export_params=True,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "time"},
        "output": {0: "batch_size", 1: "time"},
    }
)

import numpy as np
import onnxruntime

sess = onnxruntime.InferenceSession("model.onnx")
input = np.expand_dims(
    np.array([1, 2, 3], dtype=np.int64),
    0
)
output = sess.run(None, {"input": input})

from torch import nn
import torch

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(15, 96)
        enc_layer = nn.TransformerEncoderLayer(
            96,
            nhead=12,
            dim_feedforward=96,
            dropout=0.2,
            batch_first=True
        )
        self.attn_layers = nn.TransformerEncoder(
            enc_layer,
            num_layers=10,
            enable_nested_tensor=True
        )

    def forward(self, x):
        x = self.emb(x)
        return self.attn_layers(x)


model = Model().eval()
sample_input = torch.randint(0, 15, (1, 20))
# sample_input = torch.tensor([[1, 2, 3]])
exported = torch.onnx.dynamo_export(
    model,
    sample_input,
    export_options=torch.onnx.ExportOptions(dynamic_shapes=True)
)

exported.save("model.onnx")
print("Exported")

import numpy as np
import onnxruntime

session = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input = np.array([[1, 2, 3]], dtype=np.int64)
output = session.run(None, {"arg0": input})
torch_input = torch.tensor(input)
torch_output = model(torch_input)
torch.testing.assert_close(torch.tensor(output[0]), torch_output)
print(output)