import torchvision

from torch import multiprocessing as mp
import torch

def f(q):
    x = torch.rand(5, 5)
    q.put(x)  # <---- Change this to `q.put(x.numpy())` to fix
    q.close()
    q.join_thread()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    nworkers = 5
    q = mp.Queue()
    workers = [mp.Process(target=f, args=(q,)) for _ in range(nworkers)]
    [w.start() for w in workers]

    for _  in range(nworkers):
        x = q.get()
        print(x)

    [w.join() for w in workers]

import torch
import torch.multiprocessing as mp
from torchvision.models import AlexNet
import multiprocessing

def infer(rank, task_queue, result_queue, num_gpus):
    """Each subprocess will run this function on a different GPU indicated by `rank`."""
    model = AlexNet()
    device = torch.device(f"cuda:{rank % num_gpus}")
    model.to(device)
    while True:
        task = task_queue.get()
        if task is None:  # Check for sentinel value
            break
        task_index, a, b = task  # Unpack the task
        x = a + b
        x = x.to(device)
        with torch.no_grad():
            # LP: see https://github.com/pytorch/pytorch/issues/60654
            output = model(x).cpu().numpy()  # Detach the output from the computation graph and move to CPU
            result_queue.put((task_index, rank, output))  # Store the result in the result queue with task index
            del a, b  # free memory
            print(f"Inference on process {rank}, using GPU {rank % num_gpus}")

def main(num_processes, num_gpus):
    
    task_queue = mp.Queue()
    result_queue = mp.Queue()  # Queue to collect results
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=infer, args=(rank, task_queue, result_queue, num_gpus))
        p.start()
        processes.append(p)
    
    for task_index in range(10):
        a = torch.randn(1, 3, 224, 224)
        b = torch.randn(1, 3, 224, 224)
        task_queue.put((task_index, a, b))
    
    for _ in range(num_processes):
        # # sentinel value to signal subprocesses to exit
        task_queue.put(None)  # Only None to match the expected number of arguments
    
    for p in processes:
        p.join()  # wait for all subprocesses to finish
        
    # Now collect and print all the results from the result queue
    results = []
    while not result_queue.empty():
        task_index, rank, result = result_queue.get()
        results.append((task_index, rank, result))
    
    # Sort results by task index to maintain the original order
    results.sort(key=lambda x: x[0])
    
    for task_index, rank, result in results:
        print(f"Result from task {task_index}, process {rank}: {result.size}")

if __name__ == "__main__":
    # LP: Ensures the spawn start method is used, which is safer for CUDA operations in multiprocessing.
    mp.set_start_method('spawn', force=True)
    # Number of subprocesses, adjust according to your needs
    num_processes = 4
    # you may want to use
    # num_processes = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()  # Automatically get the number of GPUs available
    main(num_processes, num_gpus)