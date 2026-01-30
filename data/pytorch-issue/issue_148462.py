import os
import torch
import torch.distributed as dist

USE_ASYNC = True

def test_func(x, rank):
    if rank == 0:
        x += 1
        # Send the tensor to process 1
        if USE_ASYNC:
            a = dist.isend(tensor=x, dst=1)
        else:
            dist.send(tensor=x, dst=1)
    else:
        # Receive tensor from process 0
        if USE_ASYNC:
            a = dist.irecv(tensor=x, src=0)
        else:
            dist.recv(tensor=x, src=0)
    if USE_ASYNC:
        a.wait()
    return x + 2

def run(rank):
    torch.cuda.set_device(rank)
    x = torch.ones(1, device='cuda')
    with torch.cuda.stream(torch.cuda.Stream()):
        for i in range(11):
            x.copy_(torch.ones(1, device='cuda'))
            y = test_func(x, rank)
            print(f"Rank{rank} has data {y} in warmup")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()

    x.copy_(torch.ones(1, device='cuda'))
    with torch.cuda.graph(graph):
        y = test_func(x, rank)

    for i in range(1):
        x.copy_(torch.ones(1, device='cuda'))
        graph.replay()
    print(f"Rank{rank} has data {y} after graph replay")


def main():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    run(local_rank)

if __name__ == "__main__":
    main()