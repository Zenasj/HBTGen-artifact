import torch.nn as nn

import os
import torch
import torch.distributed as dist
torch._logging.set_logs(graph=True, graph_code=True)
class allgather_in_tensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_tensor, x):
        torch.distributed.all_gather_into_tensor(out_tensor, x)
        return out_tensor


def test_allgather_in_tensor_static(rank, world_size):
    torch.cuda.set_device("cuda:" + str(rank))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    x = torch.ones(2, 2, dtype=torch.int64).to("cuda:" + str(rank)) + 1 + 2 * rank
    print("x-----===:", x)
    tensor_list = torch.zeros(4, 2, dtype=torch.int64).to("cuda:" + str(rank))
    print("tensor_list-----===:", tensor_list)
    mod = allgather_in_tensor()
    mod = mod.to("cuda:" + str(rank))
    ori_result = mod(tensor_list, x)
    print("ori_result:", ori_result)
    torch._dynamo.reset()
    opt_mod = torch.compile(mod, dynamic=False, fullgraph=True)
    compile_result = opt_mod(tensor_list, x)
    print("compile_result:", compile_result)
    assert ori_result.equal(compile_result)
	
def mp():
    world_size = 2
    torch.multiprocessing.spawn(test_allgather_in_tensor_static, args=(world_size,), nprocs=world_size, join=True)


if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29506"
    mp()