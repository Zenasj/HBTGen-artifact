import multiprocessing as mp
try:
    mp.set_start_method('spawn') # spawn, forkserver, and fork
except RuntimeError:
    pass