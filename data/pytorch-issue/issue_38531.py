import torch.distributed.rpc as rpc
from torch.multiprocessing import Process

import os
import time

def do_nothing():
    pass

def test(rank, size):
    rpc.init_rpc("Rank"+str(rank), rank=rank, world_size=size)
    print("Rank %s rpc init" % rank)

    i = 0
    # To test easily, I changed <PATH_TO_TORCH_LIB>/torch/distributed/rpc/backend_registry.py.
    # Under def _process_group_init_backend_handler(),
    # I changed the below line
    # >> process_group_timeout = rpc_constants.DEFAULT_PROCESS_GROUP_TIMEOUT
    # (which makes the timeout 30 minutes), to somewhat shorter value, e.g.,
    # >> process_group_timeout = datetime.timedelta(seconds=10).
    # Otherwise, if I wait for 30 min the problem still occurs.

    ## Loop that does not do anything for a long time...
    while i < 10:
        time.sleep(1)
        print("Rank %s %s sec passed..." % (rank, i))

        ## Uncommenting the below two lines makes the crash go away!
        ## I.e., generating some RPC traffic.
        #target = rank ^ 0x1
        #rpc.rpc_sync("Rank"+str(target), do_nothing)
        i += 1

    rpc.shutdown()
    print("Rank %s rpc shutdown" % rank)
    pass

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "29502"

    processes = []
    for rank in [0,1]:
        p = Process(target=test, args=(rank, 2, ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()