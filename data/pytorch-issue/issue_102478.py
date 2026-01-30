import torch.nn as nn

import os
from typing import Callable
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List


def my_compiler(gm:torch.fx.GraphModule, example_inputs:List[torch.Tensor]):
    
    print(gm.graph)
    gm.graph.print_tabular()
    return gm.forward


class ConvBroadcast(torch.nn.Module):

    def __init__(self, dtype):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2, 2), dtype=dtype)

    def forward(self, input_tensor):
        output = self.conv1(input_tensor)
        # broadcast from rank 0
        dist.broadcast(self.conv1.weight, src=0)
        return output


def test_broadcast(rank: int, size: int):
    
    group = dist.new_group(list(range(size)))
    model = ConvBroadcast(dtype=torch.float32)
    compiled_model = torch.compile(model, backend=my_compiler)

    torch.manual_seed(seed=dist.get_rank())
    input_tensor = torch.rand(1,5,5, dtype=torch.float32)
    print(f' >>> Rank: {dist.get_rank()} input_tensor is\n {input_tensor}')
    compiled_model(input_tensor=input_tensor)
    print(f' >>> Rank: {dist.get_rank()} trained weights is\n {compiled_model.conv1.weight}')


def init_process (rank: int, size: int, fn: Callable[[int, int], None], backend="gloo"):

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    dist.destroy_process_group()


if  __name__ == "__main__":

    size = 2
    processes = []
    mp.set_start_method("spawn")

    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, test_broadcast))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()