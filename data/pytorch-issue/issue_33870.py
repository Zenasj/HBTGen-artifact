import torch

torch.autograd.set_detect_anomaly(True)

a = torch.rand([], requires_grad=True, device="cuda:0")
b = torch.rand(10, requires_grad=True, device="cuda:1")

c = a * b
c.sum().backward() # Fails
# RuntimeError: Function MulBackward0 returned an invalid gradient at index 0 - expected device cuda:0 but got cuda:1 (validate_outputs at /opt/conda/conda-bld/pytorch_1582704525289/work/torch/csrc/autograd/engine.cpp:477)