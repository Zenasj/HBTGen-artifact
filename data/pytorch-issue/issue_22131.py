# bug1.py
# python3 bug1.py

# Does not break if `import librosa` is removed or set_start_method is removed

import torch
import torch.utils.data
import librosa

class Dataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return torch.zeros(2, 4)

    def __len__(self):
        return 1

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force = True)
    loader = torch.utils.data.DataLoader(Dataset(), num_workers = 1)
    next(iter(loader))

#/miniconda/lib/python3.7/multiprocessing/semaphore_tracker.py:144: UserWarning: semaphore_tracker: There appear to be 1 leaked semaphores to clean up at shutdown
#  len(cache))

# bug2.py
# python3 bug2.py

if __name__ == '__main__':
    import torch
    import torch.utils.data
    import librosa

    class Dataset(torch.utils.data.Dataset):
        def __getitem__(self, index):
            return torch.zeros(2, 4)

        def __len__(self):
            return 1

    torch.multiprocessing.set_start_method('spawn', force = True)
    loader = torch.utils.data.DataLoader(Dataset(), num_workers = 1)
    next(iter(loader))

#Traceback (most recent call last):
#  File "<string>", line 1, in <module>
#  File "/miniconda/lib/python3.7/multiprocessing/spawn.py", line 105, in spawn_main
#    exitcode = _main(fd)
#  File "/miniconda/lib/python3.7/multiprocessing/spawn.py", line 115, in _main
#    self = reduction.pickle.load(from_parent)
#AttributeError: Can't get attribute 'Dataset' on <module '__mp_main__' from #'/deepspeech.pytorch/convasr/bug_.py'>
#Traceback (most recent call last):
#  File "/miniconda/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 512, in _try_get_batch
#    data = self.data_queue.get(timeout=timeout)
#  File "/miniconda/lib/python3.7/multiprocessing/queues.py", line 104, in get
#    if not self._poll(timeout):
#  File "/miniconda/lib/python3.7/multiprocessing/connection.py", line 257, in poll
#    return self._poll(timeout)
#  File "/miniconda/lib/python3.7/multiprocessing/connection.py", line 414, in _poll
#    r = wait([self], timeout)
#  File "/miniconda/lib/python3.7/multiprocessing/connection.py", line 920, in wait
#    ready = selector.select(timeout)
#  File "/miniconda/lib/python3.7/selectors.py", line 415, in select
#    fd_event_list = self._selector.poll(timeout)
#  File "/miniconda/lib/python3.7/site-packages/torch/utils/data/_utils/signal_handling.py", line 63, in handler
#    _error_if_any_worker_fails()
#RuntimeError: DataLoader worker (pid 6082) exited unexpectedly with exit code 1. Details are #lost due to multiprocessing. Rerunning with num_workers=0 may give better error trace.

#During handling of the above exception, another exception occurred:

#Traceback (most recent call last):
#  File "bug_.py", line 15, in <module>
#    next(iter(loader))
#  File "/miniconda/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 577, in #__next__
#    idx, batch = self._get_batch()
#  File "/miniconda/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 554, in #_get_batch
#    success, data = self._try_get_batch()
#  File "/miniconda/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 520, in #_try_get_batch
#    raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str))
# RuntimeError: DataLoader worker (pid(s) 6082) exited unexpectedly

import torch
import torch.utils.data

class Dataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return torch.zeros(2, 4)

    def __len__(self):
        return 1


if __name__ == '__main__':
    import librosa
    torch.multiprocessing.set_start_method('spawn', force = True)
    loader = torch.utils.data.DataLoader(Dataset(), num_workers = 1)
    next(iter(loader))