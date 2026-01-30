import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        MAXINT = 2147483647
        self.len = MAXINT + 10
        self.len = 200
        self.param = torch.Tensor([0]).cuda()
        self.params = []
        for _ in range(self.len):
            self.params.append(self.param)
            if _ % 100000 == 0:
                print(_)
        self.params = nn.ParameterList(self.params)

    def forward(self, x):
        output = x
        for i in range(self.len):
            output = output + self.params[i]
        return output


def example(rank, world_size):
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    model = MyModel()
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    # loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.Tensor([0]).to(rank))
    # labels = torch.randn(20, 10).to(rank)
    # backward pass
    # loss_fn(outputs, labels).backward()
    print("forward end, outputs:", outputs)
    outputs.backward()
    print("backward end")
    # update parameters
    optimizer.step()

    ddp_model.logger.set_runtime_stats_and_log()
    # ddp_logging_data = ddp_model._get_ddp_logging_data()
    # indices = ddp_logging_data.get("prev_iteration_grad_ready_order_indices")
    # print("prev_iteration_grad_ready_order_indices", indices)
    ddp_logging_data = ddp_model.logger._get_ddp_logging_data()
    # [:100] represents the first 100 characters
    indices = ddp_logging_data.strs_map["prev_iteration_grad_ready_order_indices"][:100]
    print("prev_iteration_grad_ready_order_indices", indices)


def main():
    world_size = 1
    mp.spawn(example,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
    print("done")