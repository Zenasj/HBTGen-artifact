import torch

def run(i, *args):
    print("I WAS SPAWNED BY:", args[0])
    
    tsr = torch.zeros(1)
    
    if args[0] == 0:
        tsr += 100
        dist.send(tsr, dst=1)
    else:
        dist.recv(tsr)
        print ("RECEIVED VALUE =", tsr)

if __name__ == '__main__':
    
    # Initialize Process Group
    dist.init_process_group(backend="mpi")
    
    mp.set_start_method('spawn')    
    
    # get current process information
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # spawn sub-processes
    mp.spawn(run, args=(rank, world_size,), nprocs=1)