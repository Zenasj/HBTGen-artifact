import torch
import torch.nn as nn
import random

cudnn.benchmark = True
cudnn.deterministic = True

transforms.RandomCrop(32, padding=4),
transforms.RandomHorizontalFlip(),

cudnn.benchmark = False
cudnn.deterministic = True

random.seed(1)
numpy.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

def worker_init(worker_id):
    random.seed(args.base_seed)

def _worker_loop(dataset, index_queue, data_queue, collate_fn, seed, init_fn, worker_id):
    global _use_shared_memory
    _use_shared_memory = True

    # Intialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
    # module's handlers are executed after Python returns from C low-level
    # handlers, likely when the same fatal signal happened again already.
    # https://docs.python.org/3/library/signal.html Sec. 18.8.1.1
    _set_worker_signal_handlers()

    torch.set_num_threads(1)
    random.seed(seed)
    torch.manual_seed(seed)

    if init_fn is not None:
        init_fn(worker_id)