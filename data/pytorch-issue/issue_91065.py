import torch
a = torch.rand(5, dtype=torch.float32, device="cpu")
ret = a.unsafe_chunk(4, 0)
y = ret[0] * ret[2]
a_mps = a.to("mps")
ret_mps = a_mps.unsafe_chunk(4, 0)
y_mps = ret_mps[0] * ret_mps[2]
torch.testing.assert_close(y, y_mps, check_device=False)