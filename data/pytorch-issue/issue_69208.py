import torch
x = torch.randn(2, 2, 2, 2, device="cuda")
w = torch.randn(2, 2, 2, device="cuda")
b = torch.zeros(2, 2, 2, device="cuda")  # zero bias shouldn't affect output values.
o1 = torch.layer_norm(x, [2, 2, 2], w, b, False)
o2 = torch.layer_norm(x, [2, 2, 2], w, None, False)
o1.allclose(o2)  # yet it does!