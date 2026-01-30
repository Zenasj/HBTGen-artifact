py
import torch

class Dataset(object):
    def __getitem__(self, index):
        return torch.tensor(index, device='cuda')

    def __len__(self):
        return 10


if __name__ == '__main__':
    dataloader = torch.utils.data.DataLoader(
        Dataset(), multiprocessing_context='spawn', num_workers=2)
    for x in dataloader:
        print(x)

def _worker_loop(queue, collate_fn, need_shutdown, data_loader_iterator, batch_size, drop_last):
    def _put(batch):
        if not isinstance(batch, StopIteration):
            batch = collate_fn(batch)
        while True:
            try:
                queue.put(batch, block=True, timeout=1)
                return True
            except Queue.Full:
                if need_shutdown.is_set():
                    return False
    batch = []
    for sample in data_loader_iterator:
        sample = copy.deepcopy(sample)
        batch.append(sample)
        if len(batch) == batch_size:
            if not _put(batch):
                # Need to shutdown
                return
            batch = []
    if len(batch) > 0 and not drop_last:
        if not _put(batch):
            # Need to shutdown
            return
    # Signal the end of the queue
    if not _put(StopIteration()):
        # Need to shutdown
        return

import torch
# Hack for fork-safe issue of PyTorch, see https://github.com/pytorch/pytorch/pull/25158, to be removed after torch
# fixes this in upstream.
# Start of torch DataLoader iterator fork-safe hack
import threading
from multiprocessing.util import register_after_fork
from torch.multiprocessing.reductions import SharedCache


class _SharedCache(SharedCache):
    def __init__(self):
        self.limit = 128
        self._after_fork()
        register_after_fork(self, _SharedCache._after_fork)

    def _after_fork(self):
        self.lock = threading.Lock()


torch.multiprocessing.reductions.shared_cache = _SharedCache()
# End of torch DataLoader iterator fork-safe hack