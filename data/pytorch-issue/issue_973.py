import torch
import torch.multiprocessing as multiprocessing

def create_one(batch_size):
    return torch.ByteTensor(batch_size, 128, 128)

pool = torch.multiprocessing.Pool(torch.multiprocessing.cpu_count())
data = pool.map(create_one, torch.LongTensor(1000).fill_(100))

import torch
import torch.multiprocessing as multiprocessing

def _worker_loop(data_queue, ):
    while True:
        t = torch.FloatTensor(1)
        data_queue.put(t)


if __name__ == '__main__':
    data_queue = multiprocessing.Queue(maxsize=1)
    p = multiprocessing.Process(
        target=_worker_loop,
        args=(data_queue,))

    p.daemon = True
    p.start()
    lis = []
    for i in range(10000):
        try:
            lis.append(data_queue.get())
        except:
            print('i = {}'.format(i))
            raise

from copy import deepcopy

pred_list = []
target_list = []
# long version
for inputs, targets in DataLoader(dataset, num_workers=6, batch_size=64):
    pred_list.append(model.predict_on_batch(inputs))  # make model prediction
    targets_copy = deepcopy(targets)
    target_list.append(targets_copy)
    del inputs
    del targets

# short version. interpreter invokes `del` automatically for inputs and targets
for inputs, targets in DataLoader(dataset, num_workers=6, batch_size=64):
    pred_list.append(model.predict_on_batch(inputs))  # make model prediction
    target_list.append(deepcopy(targets))  # append targets copy. `targets` removed automatically