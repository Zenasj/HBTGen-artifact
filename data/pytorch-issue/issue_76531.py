import torch

K1 = torch.tensor([[-0.3329024952074725,  0.0,  0.0,  0.0],
        [ 0.0019572051122251537, -0.3244421706868837,  0.00000,  0.00000],
        [-0.0016241708944436939, -0.007378259691806472, -0.3272105439699563,  0.00000],
        [ 0.023828483586765488,  0.10824768536769169, -0.08982846962075508,  0.9845552098643126]], device='cuda:0', dtype=torch.float64, requires_grad=True)

g1 = torch.tensor([[ 0.0,  0.0,  0.0,  7.273988740053028e-05],
        [ 0.0,  0.0,  0.0, -0.00011222996545257047],
        [ 0.0,  0.0,  0.0, -0.0005098145338706672],
        [ 0.0,  0.0,  0.0,  0.00042305406532250345]], device='cuda:0', dtype=torch.float64)

print(torch.linalg.eigh(K1)[1].grad_fn(None, g1))
# tensor([[nan, nan, nan, nan],
#         [nan, nan, nan, nan],
#         [nan, nan, nan, nan],
#         [nan, nan, nan, nan]], device='cuda:0', dtype=torch.float64,
#        grad_fn=<MmBackward0>)