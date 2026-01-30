import torch
from torch_geometric.utils import scatter

w = torch.load(r'test_tensor.pth')
mu_ConnG = w["mu"].to("cuda")
batch = w["index"].to("cuda")
y = scatter(mu_ConnG, index=batch, dim=0,  reduce="max")

mu_ConnG = torch.Tensor([
    [[1e-3, 2, 3],
     [1, 5, 6]],

    [[4e-4, 8, 9],
     [10, 11, 12]],

    [[13, 14, 15],
     [18, 17, 16]],

    [[100, 8, 9],
     [10, 19, 12]],

    [[100, 8, 9],
     [99, 19, 12]],

    [[4583, 456, 9],
     [10, 19, 1]],
]).double().to("cuda")

batch = torch.LongTensor([0, 0, 1, 1, 2, 0]).to("cuda")
y1 = scatter(mu_ConnG, index=batch, dim=0,  reduce="max")
y2 = scatter_max(mu_ConnG, index=batch, dim=0)

tensor([[[4583.,  456.,    9.],
         [  10.,   19.,   12.]],

        [[ 100.,   14.,   15.],
         [  18.,   19.,   16.]],

        [[ 100.,    8.,    9.],
         [  99.,   19.,   12.]]], device='cuda:0', dtype=torch.float64)