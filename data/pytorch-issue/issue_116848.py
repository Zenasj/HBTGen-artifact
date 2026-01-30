import torch.nn as nn

# TORCH_COMPILE_DEBUG=1 python -u check_issue_upsample_nearest.py

import torch
import torch.nn.functional as F

class DWModel(torch.nn.Module):
    def __init__(self):
        super(DWModel, self).__init__()

    def forward(self, hidden_states):
        return F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        # return F.interpolate(hidden_states, scale_factor=0.5, mode="nearest")


torch.manual_seed(1234321)
model = DWModel()
# cmodel = torch.compile(model, backend="eager")   # this option does not fail for all test cases 
cmodel = torch.compile(model, backend="inductor")

dtype = torch.float32

torch.manual_seed(1234321)
# input = torch.rand((3, 640, 32, 32), device='cuda', dtype=dtype)
# input = torch.rand((3, 640, 32, 32), device='cpu', dtype=dtype)
# input = torch.ones((3, 640, 32, 32), device='cuda', dtype=dtype)
# input = torch.ones((3, 640, 32, 32), device='cpu', dtype=dtype)
# input = torch.ones((3, 640, 4, 4), device='cpu', dtype=dtype)
input = torch.arange(3 * 640 * 4 * 4, device='cpu', dtype=dtype).reshape(3, 640, 4, 4)

expected = model(input)
output = cmodel(input)

torch.testing.assert_close(expected, output)

with torch.no_grad():
    expected2 = model(input)

with torch.no_grad():
    output2 = cmodel(input)

assert not expected2.requires_grad
assert not output2.requires_grad
torch.testing.assert_close(expected2, output2)

with torch.inference_mode():
    expected3 = model(input)

with torch.inference_mode():
    output3 = cmodel(input)

assert not expected3.requires_grad
assert not output3.requires_grad
# torch.testing.assert_close(expected2, expected3)

print("mean expected3:", expected3.shape, expected3.mean())
print("mean output3:", output3.shape, output3.mean())

m = expected3 != output3
print("N:", m.sum())
print("expected3:", expected3[m][:10])
print("output3:", output3[m][:10])
# print(torch.argwhere(m))

torch.testing.assert_close(expected3, output3)