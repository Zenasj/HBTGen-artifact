import torch.nn as nn
import torch
import torch.multiprocessing as mp


import torchvision
import torch.distributed as dist





def main():
    mp.spawn(main_worker, nprocs=2,
                    args=(2, 1))

def main_worker(rank,process_num,nothing):
    dist.init_process_group(backend="gloo", init_method='tcp://127.0.0.1:1233',
                                    world_size=2, rank=rank)
    print("rank",rank)
    if rank == 0:
        
        input = torch.rand([4,4])
        dist.isend(input,1)


     
     

    else:
        input = torch.rand([4,4])
        dist.recv(input,0)
    print("finish",rank)
 
 
if __name__ == "__main__":
    main()

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def main():
    mp.spawn(main_worker, nprocs=2, args=(2,))
    



def main_worker(rank,n):
    print("begin 1")
    dist.init_process_group(backend = 'gloo', init_method='tcp://127.0.0.1:23456',
                        rank=rank, world_size=2)
    print(rank)
    
    # for i in range(10):
    if rank == 1:
        input = torch.rand([10,10])
        # for i in range(10):
        #     dist.recv(input,0)
        #     print(i)
        dist.recv(input,0)
        print("finish recv")
    else:
        input = torch.rand([10,10])
        dist.isend(input,1)
        print("finish send")
    dist.barrier()

if __name__ == '__main__':
    main()