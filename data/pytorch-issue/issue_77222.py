import torch.nn as nn

import torch
from time import time

def farthest_point_sample(xyz, npoint):
    # https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/eb64fe0b4c24055559cea26299cb485dcb43d8dd/models/pointnet2_utils.py#L63-L84
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.zeros(B, dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

class FarthestPointSample(torch.nn.Module):
    def __init__(self):
        super(FarthestPointSample, self).__init__()

    def forward(self, xyz, points):
        return farthest_point_sample(xyz, points)

fmodel = FarthestPointSample()
dummy_input = torch.rand(1, 1024, 3)
print('test running the model:')
output = fmodel(dummy_input, 100)
print(output)
print('test export sa1 model')
time_values = []
for x in [50, 100, 150, 200, 250]:
    t1 = time()
    torch.onnx.export(fmodel, (dummy_input, x), '/tmp/dummy_model.onnx', verbose=True)
    time_values.append(time()-t1)
    print(time_values)
print('Finished fmodel')