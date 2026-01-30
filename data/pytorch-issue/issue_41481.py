import torch
import torch.multiprocessing as mp    
import time

def main_worker(rank, share_memory_resources):
    print(f'rank={rank}')
    print(share_memory_resources['X'][0][:5])
    print(share_memory_resources['y'][0][:5])
    while True:
        time.sleep(5)
    
def main():
    world_size = 4
    n, m =  10, 360
    n_keys = 3000    # <---------- different number of `n_keys` gives different results
    #  3000: RuntimeError: unable to open shared memory object </torch_71861_711505626> in read-write mode
    #  500: OSError: [Errno 24] Too many open files
    #  100: OK

    share_memory_resources = {
        'X': {
            i: torch.rand(size=(n, m)) for i in range(n_keys)
        },
        'y': {
            i: torch.rand(size=(n, m)) for i in range(n_keys)
        }
    }

    # calculate memory_usage. Optional
    total_memory_usage = 0.0
    for k, inner_dict_ in share_memory_resources.items():
        for k_, v_ in inner_dict_.items():
            total_memory_usage += v_.data.numpy().nbytes
    print(f'total_memory_usage = {total_memory_usage / 1024 ** 2} MB')
    
    mp.spawn(main_worker, args=(share_memory_resources, ), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

mp.set_sharing_strategy('file_system')