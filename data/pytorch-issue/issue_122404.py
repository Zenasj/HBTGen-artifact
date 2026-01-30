import os
import torch
import torch.distributed as dist
from bug import foo


def init_dist():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", "54321"))
    init_method = f"tcp://{master_addr}:{master_port}"
    print(f"Init dist: {init_method}, rank {rank}/{world_size}")
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world_size)


def main():
    init_dist()
    device = torch.device("cuda", dist.get_rank() % torch.cuda.device_count())

    torch.manual_seed(42 + dist.get_rank())
    x = torch.randn(128, 256, device=device)
    y = torch.compile(foo)(x)
    print(y)

if __name__ == "__main__":
    main()

# import torch
import torch.distributed as dist
from torch import Tensor


def foo(x: Tensor) -> Tensor:
    m = x.max()
    dist.all_reduce(m, op=dist.ReduceOp.MAX)
    return m

source = AttrSource(
    AttrSource(
        base=AttrSource(
            base=GlobalSource(global_name="torch"),
            member="distributed",
            get_static=False,
        ),
        member="group",
        get_static=False,
    ),
    member="WORLD",
    get_static=False,
)
install_guard(source.make_guard(GuardBuilder.ID_MATCH))