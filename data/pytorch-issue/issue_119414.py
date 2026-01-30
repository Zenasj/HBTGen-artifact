import logging
from torch._logging import set_logs
import torch

torch._dynamo.config.capture_scalar_outputs = True

set_logs(
    dynamo=logging.DEBUG,
    aot=logging.DEBUG,
    inductor=logging.DEBUG,
    ddp_graphs=True,
    graph_breaks=True,
    guards=True,
    recompiles=True,
    dynamic=logging.INFO,
)

def repro(length_per_key, values, weights):
    split_points = length_per_key.tolist()
    torch._check(sum(split_points) == values.shape[0])
    torch._check(sum(split_points) == weights.shape[0])
    split = torch.functional.split(values, split_points)
    split_1 = torch.functional.split(weights, split_points)
    return split, split_1

weights = torch.rand(35, device='cuda')
values = torch.rand(35, device='cuda')
length_per_key = torch.tensor([17,17,1], device='cuda')

compiled = torch.compile(repro)

compiled(length_per_key, values, weights)