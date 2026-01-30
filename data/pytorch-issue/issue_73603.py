import os
import pickle as pkl 
import numpy as np
import torch

class Simple_Dataset(torch.utils.data.IterableDataset):
    def __init__(self, bsz):
        super(Simple_Dataset, self).__init__()
        self.batch_size = bsz 
        self.num_examples_per_task = 10 * self.batch_size

    def __len__(self):
        # Returns the len of dataset on the task process
        return self.num_batch_per_task*self.batch_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        wrk_id = worker_info.id
        for ind in range(self.num_examples_per_task):
            features = {}
            features["dat"] = self.num_examples_per_task * wrk_id + ind 
            features["node_id"] = wrk_id
            yield features


BATCH_SIZE=64
for RUN in range(1,4):
    print('RUN %d' % RUN)

    dataset = Simple_Dataset(BATCH_SIZE)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=8,
        drop_last=True,
        prefetch_factor=10,
        persistent_workers=True,
    )   
    for epoch in range(10):
        print('epoch = %d' % epoch)
        for batch_ind, dat in enumerate(dataloader):
            datdat = dat['dat'].numpy()
            worker = dat['node_id'].numpy()
            # only checking the first batch element in both epochs
            if batch_ind == 0:
                assert np.all(worker == 0), 'Worker id not == 0  [%s]' % str(np.unique(worker))
                assert np.all(np.diff(datdat) == 1), 'step size != 1  [%s]' %np.unique(np.diff(datdat))
                assert datdat[0] == 0, 'first lement not 0  [%d]'  %datdat[0]

self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))