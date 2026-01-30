import torch

device = xm.xla_device()

a = torch.rand(3, device=device)
b = a.to(torch.float64)

# First, 'a' and 'b' are shown, as expected, as f32 and f64 data-types, respectively.
print("[bef] > a:", torch_xla._XLAC._get_xla_tensors_text([a]))
print("[bef] > b:", torch_xla._XLAC._get_xla_tensors_text([b]))

a.data = b

# But, then, we see that nothing changed, even after setting 'a.data'!
print("[aft] > a:", torch_xla._XLAC._get_xla_tensors_text([a]))
print("[aft] > b:", torch_xla._XLAC._get_xla_tensors_text([b]))