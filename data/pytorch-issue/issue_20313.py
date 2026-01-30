import torch

def setup_process(rank, world_size, backend='gloo'):
    """
    Initialize the distributed environment (for each process).

    gloo: is a collective communications library (https://github.com/facebookincubator/gloo). My understanding is that
    it's a library/API for process to communicate/coordinate with each other/master. It's a backend library.
    """
    if rank != -1:  # -1 rank indicates serial code
        # set up the master's ip address so this child process can coordinate
        # os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12355'
        os.environ['MASTER_PORT'] = '92345'

        # - use NCCL if you are using gpus: https://pytorch.org/tutorials/intermediate/dist_tuto.html#communication-backends
        if torch.cuda.is_available():
            backend = 'nccl'
        # Initializes the default distributed process group, and this will also initialize the distributed package.
        dist.init_process_group(backend, rank=rank, world_size=world_size)