import torch

x = torch.rand([4, 4])

with enable_torch_dispatch_mode(FakeTensorMode):
     # conversion from x to Meta/Fake cached for duration of the `FakeTensorMode` call
     y = x[0]
     z = x[1]

self.assertEqual(torch._C._storage_id(y), torch._C._storage_id(z))

x = torch.rand([4, 4])

with enable_torch_dispatch_mode(FakeTensorMode):
     # conversion from x to Meta/Fake cached for duration of the `FakeTensorMode` call
     y = x[0]