py
import time
import os
import threading
from multiprocessing import Pool


WORLD_SIZE = 10000

import torch.distributed as dist

def run(rank):
    should_log = rank % (WORLD_SIZE // 10) == 0
    if should_log:
        print(f"started {rank}")
    store = dist.TCPStore(
        host_name="devvm4382.nao0.facebook.com",
        port=29500,
        world_size=WORLD_SIZE,
        is_master=rank == 0,
        use_libuv=True,
    )
    if should_log:
        print(f"init {rank}")
    store.set(f"key{rank}", "1234")
    if should_log:
        print(f"set {rank}")
    del store

def noop(rank):
    pass


print("starting pool")
with Pool(WORLD_SIZE) as pool:
    pool.map(noop, range(WORLD_SIZE), 1)
    print("pool hot")
    start = time.time()
    pool.map(run, range(WORLD_SIZE), 1)
    print("run finished", time.time()-start)