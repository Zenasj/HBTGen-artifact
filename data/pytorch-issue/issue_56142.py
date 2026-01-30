import os
import torch

class Dummy:
    def __init__(self):
        self.test_int = torch.distributed.get_rank()

def main():
    dummy = Dummy()

    print("Before broadcasting...")
    print(f"[RANK {torch.distributed.get_rank()}] test_int = {dummy.test_int}")

    torch.distributed.broadcast_object_list([dummy], src=0)

    print("After broadcasting...")
    print(f"[RANK {torch.distributed.get_rank()}] test_int = {dummy.test_int}")

if __name__ == "__main__":
    torch.distributed.init_process_group(backend='gloo', init_method='env://')
    main()

torch.distributed.broadcast_object_list([dummy], src=0)

to_broadcast = [dummy]
dist.broadcast_object_list(to_broadcast, src=0)
dummy = to_broadcast[0]