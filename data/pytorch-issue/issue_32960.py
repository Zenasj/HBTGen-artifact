import torch.nn as nn

import onnx
import onnx.numpy_helper
import torch
import onnxruntime as rt


class Model(torch.nn.Module):
    def forward(self, data, indices, updates):
        return torch.scatter_add(data, 0, indices, updates)

model = Model()
data = torch.tensor([0.0])
indices = torch.tensor([0, 0, 0])
updates = torch.tensor([1.2, 2.3, 3.4])
# This works.
# indices = torch.tensor([0])
# updates = torch.tensor([1.2])
torch.onnx.export(model, (data, indices, updates), 'scatter_add.onnx')

m = onnx.load('scatter_add.onnx')
sess = rt.InferenceSession('scatter_add.onnx')
inputs = {m.graph.input[0].name: data.numpy(),
          m.graph.input[1].name: indices.numpy(),
          m.graph.input[2].name: updates.numpy()}
assert sess.run([m.graph.output[0].name], inputs)[0] == model(data, indices, updates).numpy()

data = torch.tensor([0.0])
indices = torch.tensor([0, 0, 0])
updates = torch.tensor([1.2, 2.3, 3.4])
print(torch.scatter_add(data, 0, indices, updates))  # [6.9]