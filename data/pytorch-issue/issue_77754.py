import torchvision
import random

import torch
import numpy as np
from torchvision.models.resnet import ResNet, BasicBlock
from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.distances import CosineSimilarity

device = torch.device("mps")

model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=64).to(device)
distance = CosineSimilarity()

miner = miners.MultiSimilarityMiner()
loss_func = losses.SoftTripleLoss(
    num_classes=10,
    embedding_size=64,
    distance=distance,
).to(device)

X, y = torch.rand(16, 3, 64, 64).to(device), torch.from_numpy(np.random.choice(np.arange(10), size=16)).to(device)
embeddings = model(X)

# NotImplementedError: Could not run 'aten::bitwise_xor.Tensor_out' with arguments from the 'MPS' backend
hard_pairs = miner(embeddings, y)
loss = loss_func(embeddings, y, hard_pairs)

# NotImplementedError: Could not run 'aten::_index_put_impl_' with arguments from the 'MPS' backend
loss = loss_func(embeddings, y)

mps_device = torch.device("mps")
z = torch.ones(5, device=mps_device)

import torch

dist = torch.distributions.Categorical(torch.tensor([0.5, 0.5]).to('mps'))
print(dist.sample())