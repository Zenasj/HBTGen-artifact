import torch.multiprocessing as mp

def main_worker(gpu, queue, event):
    print(f'gpu {gpu} putting into queue')
    queue.put({'gpu': gpu})

    print(f'gpu {gpu} waiting')
    event.wait()

def main():
    num_gpus = 4

    queue = mp.Queue()
    event = mp.Event()

    jobs = []
    for i in range(num_gpus):
        p = mp.Process(target=main_worker, args=(i, queue, event))
        p.start()
        jobs.append(p)

    print('started processes')

    for i in range(num_gpus):
        print(f'getting {i}th queue value')
        d = queue.get()
        print(d)

    event.set()
    for p in jobs:
        p.join()

main()

import multiprocessing as mp

def main_worker(gpu, queue, event):
    print(f'gpu {gpu} putting into queue')
    queue.put({'gpu': gpu})

    print(f'gpu {gpu} waiting')
    event.wait()

def main():
    num_gpus = 4

    queue = mp.Queue()
    event = mp.Event()

    jobs = []
    for i in range(num_gpus):
        p = mp.Process(target=main_worker, args=(i, queue, event))
        p.start()
        jobs.append(p)

    print('started processes')

    for i in range(num_gpus):
        print(f'getting {i}th queue value')
        d = queue.get()
        print(d)

    event.set()
    for p in jobs:
        p.join()

main()