import torch.nn as nn

import torch

class SliceModel(torch.nn.Module):
    def forward(self, data, start, end):
        return data[:, :, start:end]

data = torch.randn((1,40,41))

# Index using 1D tensors
start_index = torch.tensor([36], dtype=torch.int64)
end_index = torch.tensor([41], dtype=torch.int64)

# Using this instead works.
# start_index = torch.tensor(36, dtype=torch.int64)
# end_index = torch.tensor(41, dtype=torch.int64)

model = SliceModel()
print("Forward function succeeds: ", model(data, start_index, end_index).shape)

# Exported model fails with Onnx Runtime
torch.onnx.export(model, (data, start_index, end_index), "slice.onnx", opset_version=18,)