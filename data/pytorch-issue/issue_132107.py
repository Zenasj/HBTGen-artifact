import torch

cfn = torch.compile(torch.cumsum)

for n in [100, 10, 100]:
    print(torch.sum(cfn(torch.full((n,), float('inf'), device='cuda', dtype=torch.float64), -1)))

tensor(inf, device='cuda:0', dtype=torch.float64)
tensor(inf, device='cuda:0', dtype=torch.float64)
tensor(nan, device='cuda:0', dtype=torch.float64)

tmp4, = tl.associative_scan((tmp2,), 1, _triton_helper_fn_add0)
tl.reduce((tmp4 * (rbase == (RBLOCK - 1))), -1, _triton_helper_fn_add0, keep_dims=True)