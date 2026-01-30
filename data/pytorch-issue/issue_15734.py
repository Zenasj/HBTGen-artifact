import torch.multiprocessing as multiprocessing
import torch

def mp_worker(gpu):
    print(torch.cuda.get_device_properties(gpu))

if __name__ == '__main__':
    gpus = list(range(torch.cuda.device_count()))

    ctx = multiprocessing.get_context('spawn')

    processes = [ctx.Process(target=mp_worker, args=(gpui,)) for gpui in gpus]
    for process in processes:
        process.start()
    for process in processes:
        process.join()