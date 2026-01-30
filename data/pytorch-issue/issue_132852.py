import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import os
from torch.profiler import schedule, ProfilerActivity, tensorboard_trace_handler
class RingComm:
    """
    P2P communicator for double ring zigzag flash attn.
    """

    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)
            # print(f'rank:{self.rank},send_rank:{self.send_rank},recv_rank:{self.recv_rank}')

    def send_recv(self, to_send: torch.Tensor, recv_tensor: torch.Tensor = None) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

def init_process_group(rank=0, world_size=1):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    

def run_test(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # or the IP of the master node
    os.environ['MASTER_PORT'] = '8125'       # make sure this port is free
    init_process_group(rank, world_size)
    bs, seqlen, head_num, head_dim = 4, 4096, 32, 128
    
    qkv = torch.randn(bs, seqlen, 3, head_num, head_dim, device=f'cuda:{rank}', requires_grad=True, dtype=torch.bfloat16)
    groups = [dist.new_group() for _ in range(3)]
    with torch.profiler.profile(
            record_shapes=True,
            with_stack=True,
            with_modules=True,
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler("test"),
            schedule=schedule(wait=0, warmup=1, active=3)
    ) as prof:
        for i in range(5):      
            comm1 = RingComm(groups[0])
            comm2 = RingComm(groups[1])
            comm3 = RingComm(groups[2])
            tensor1 = comm1.send_recv(qkv)
            comm1.commit()
            tensor2 = comm2.send_recv(qkv)
            comm2.commit()
            tensor3 = comm3.send_recv(qkv)
            comm3.commit()
            
            comm1.wait()
            comm2.wait()
            comm3.wait()
            prof.step()
            
if __name__ == "__main__":
    world_size = 4
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)