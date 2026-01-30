import os

def print_layer(prefix):
    print(f'{prefix}: {os.environ.get("MKL_THREADING_LAYER")}')

if __name__ == '__main__':
    print_layer('Pre-import')
    import numpy as np
    from torch import multiprocessing as mp
    print_layer('Post-import')

    mp.set_start_method('spawn')
    p = mp.Process(target=print_layer, args=('Child',))
    p.start()
    p.join()

def child():
    import torch
    torch.set_num_threads(1)

if __name__ == '__main__':
    import mkl
    from torch import multiprocessing as mp

    mp.set_start_method('spawn')
    p = mp.Process(target=child)
    p.start()
    p.join()