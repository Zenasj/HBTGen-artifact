@onlyNativeDeviceTypes
@dtypes(torch.float)
def test_grad_scaling_unscale_sparse(self, device, dtype):
    device = torch.device(device)
    i = torch.tensor([[0, 1, 1],
                      [2, 0, 2]], device=device, dtype=torch.int64)
    v = torch.tensor([16., 32., 64.], device=device, dtype=torch.float)
    s = torch.sparse_coo_tensor(i, v, torch.Size([2, 3]), device=device, dtype=dtype)

import torch
def compile_test():
    device = torch.device("cpu")
    i = torch.tensor([[0, 1, 1],
                        [2, 0, 2]], device=device, dtype=torch.int64)
    v = torch.tensor([16., 32., 64.], device=device, dtype=torch.float)
    s = torch.sparse_coo_tensor(i, v, torch.Size([2, 3]), device=device, dtype=torch.float)

my_fn = torch.torch._dynamo.optimize("eager", nopython=False)(compile_test)

my_fn()