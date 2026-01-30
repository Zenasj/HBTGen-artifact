import torch.distributed.rpc as rpc
import os
import time

class Test:
    def __init__(self):
        self.count = 0

    def hello(self):
        return None

def test_rpcs():
    rpc.init_rpc("TestNode", rank=0, world_size=1)
    print("RPC initialized.")
    target = Test()
    rpc.rpc_async("TestNode", target.hello)
    print("RPC call completed.")
    rpc.shutdown() # <---- ADDED HERE

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "29502"
    test_rpcs()