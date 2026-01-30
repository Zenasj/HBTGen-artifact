import torch.distributed as dist
import torch.multiprocessing as mp
import torch
import os
import time

def run(rank, size):
    print(f"[ {os.getpid()} ] world_size = {dist.get_world_size()}, " + f"rank = {dist.get_rank()}, backend={dist.get_backend()}")
    time.sleep(1000)

def init_process(rank, size, fn, backend="gloo"): 
    os.environ['MASTER_ADDR'] = '192.168.0.3' 
    os.environ['MASTER_PORT'] = '29522'
    os.environ['WORLD_SIZE'] = str(size) 
    os.environ['RANK'] = str(rank)
    print(f"Pre Complete initialization {rank} {size}")
    from datetime import timedelta
    server_store = dist.TCPStore("192.168.0.3", 12345, 2, True, timedelta(seconds=30))
    server_store.set("first_key", "first_value")
    dist.init_process_group(backend, store=server_store, rank=rank, world_size=size)
    
    print(f"Complete initialization")
    fn(rank, size)

if __name__ == "__main__":
    size = 2
    processes = []
    p = mp.Process(target=init_process, args=(0, size, run))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()

import torch.distributed as dist
import torch.multiprocessing as mp
import torch
import os
import time

def run(rank, size):
    print(f"[ {os.getpid()} ] world_size = {dist.get_world_size()}, " + f"rank = {dist.get_rank()}, backend={dist.get_backend()}")

def init_process(rank, size, fn, backend="gloo"): 
    os.environ['MASTER_ADDR'] = '192.168.0.3' 
    os.environ['MASTER_PORT'] = '29522'
    os.environ['WORLD_SIZE'] = str(size) 
    os.environ['RANK'] = str(rank)
    print(f"Pre Complete initialization {rank} {size}")
    client_store = dist.TCPStore("192.168.0.3", 12345, 2, False)
    print(client_store.get("first_key"))
    dist.init_process_group(backend, rank=rank, store=client_store, world_size=size)
    
    print(f"Complete initialization")
    fn(rank, size)

if __name__ == "__main__":
    size = 2
    processes = []
    p = mp.Process(target=init_process, args=(1, size, run))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()