processes = []
for i in range(2):
    p = mp.Process(target=_run_process, args=(i, (i + 1) % 2, file_name))
    p.start()

for p in processes:
    p.join()

import torch.multiprocessing as mp
nprocs = 2
mp.spawn(_run_process, args=(nprocs, file_name), nprocs=nprocs)