import torch

for device in ["cuda", "meta"]:
    a = torch.empty_strided([4, 576, 768], (442368, 1, 576), device=device)
    b = [768]
    c = torch.empty([768], device=device)
    d = torch.empty([768], device=device)
    e = .01
    args = (a, b, c, d, e)
    print(device)
    torch.ops.aten.native_layer_norm(*args)
    print("succeeds")