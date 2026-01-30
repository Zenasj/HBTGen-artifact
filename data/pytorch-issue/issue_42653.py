def torch_pool(inputs, target_size):
      start_points = (torch.arange(target_size, dtype=torch.float32) * (inputs.size(-1) / target_size)).long()
      end_points = ((torch.arange(target_size, dtype=torch.float32)+1) * (inputs.size(-1) / target_size)).ceil().long()
      pooled = []
      for idx in range(target_size):
          pooled.append(torch.mean(inputs[:, :, start_points[idx]:end_points[idx]], dim=-1, keepdim=False))
      pooled = torch.cat(pooled, -1)
      return pooled

import torch
import torchvision
import torch.nn as nn

model = torchvision.models.vgg11()

input_shape = 224
input_shape = 512

x = torch.rand((1, 3, input_shape, input_shape))

y = model(x)
print("Got test output okay!")

torch.onnx.export(model, x, "/tmp/model.onnx")
print("Exported to ONNX")