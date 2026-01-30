import torch.nn as nn

import torch

class SimpleModel(torch.nn.Module):
    def forward(self, x):
        x = torch.arctan(x)
        x = torch.linalg.cond(x)
        return x

model = SimpleModel()
inputs = torch.ones(2, 2, dtype=torch.float32)
res = model(inputs)

compiled_model = torch.compile(model, backend='inductor')
with torch.no_grad():
    compiled_out = compiled_model(inputs)
print(res)
print(compiled_out)
non_nan_mask = ~torch.isnan(res)
torch.testing.assert_close(res[non_nan_mask], compiled_out[non_nan_mask])