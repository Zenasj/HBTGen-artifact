import torch
from torch.multiprocessing import Process, Queue, set_start_method

def worker(input_queue, results_queue, exit_queue):
    start, end, tensor = input_queue.get()
    # I put start, end too as an identifier for this process/result
    results_queue.put((start, end, tensor[start:end].float().mean()))
    exit_queue.get()

if __name__ == "__main__":
    set_start_method("spawn")

    input_queue = Queue()
    results_queue = Queue()
    exit_queue = Queue()

    tensor = torch.arange(100)
    step = 10
    n_procs = 0
    for start in range(0, len(tensor), step):
        end = start + step - 1
        input_queue.put((start, end, tensor))
        n_procs += 1

    procs = []
    for _ in range(n_procs):
        proc = Process(target=worker, args=(input_queue, results_queue, exit_queue))
        proc.start()
        procs.append(proc)

    results = [results_queue.get() for _ in range(n_procs)]
    print(results)

    for _ in range(n_procs):
        exit_queue.put(1)

    for proc in procs:
        proc.join()

def kernel(q,  device):
    batch_tensor = torch.FloatTensor(2, 1000)
    # some operation
    q.put(batch_tensor)
    print('finish')

if __name__ == "__main__":
    q = mp.Queue()
    jobs = []
    device_list = [0, 1]
    for device in device_list:
        proc = mp.Process(
            target=kernel, 
            args=(q, device)
        )
        proc.start()
        jobs.append(proc)
    for job in jobs:
        job.join()
    for device in device_list:
        batch_tensor = q.get()