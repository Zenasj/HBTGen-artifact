import torch

tensor_list = torch.tensor([1.2, 1.0], device="mps")

for scalar in tensor_list:
  r_mps = torch.ceil(scalar)
  r_cpu = torch.ceil(scalar.to("cpu"))
  self.assertEqual(r_mps.cpu(), r_cpu)

x = torch.tensor([1.0, 0.49], device="mps")
print(x) # prints 1.0 and 0.0