import torch.nn as nn

import torch
results = dict()
kernel_size = 3
stride = 1

max_pool = torch.nn.MaxPool1d(kernel_size, stride=stride, )

temp_tensor = torch.rand([17, 0, 50], dtype=torch.float32)

t1 = temp_tensor.clone().detach()
results["res_1"] = max_pool(t1)

try:
    t2 = temp_tensor.clone().detach().requires_grad_()
    results["res_2_grad"] = max_pool(t2)
except Exception as e:
    results["err_2_grad"] = str(e)

print(results)
# {'res_1': tensor([], size=(17, 0, 48)), 'err_2_grad': 'Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for input, but got:[17, 0, 1, 50]'}