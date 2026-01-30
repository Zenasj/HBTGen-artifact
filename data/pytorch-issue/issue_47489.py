import torch.multiprocessing as mp
mp.spawn(
        self._process,
        nprocs=self.gpu_nums)