import torch
import math
from torch.utils.data import DataLoader


class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))


# Define a `worker_init_fn` that configures each dataset copy differently
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


if __name__ == '__main__':
    ds = MyIterableDataset(start=0, end=500)

    dl = DataLoader(
        dataset=ds, batch_size=100, num_workers=2, worker_init_fn=worker_init_fn,
    )

    for e in dl:
        print(e.shape)

batches_per_worker = (overall_end - overall_start) // batch_size // num_workers
examples_per_worker = batches_per_worker * batch_size

def worker_init_fn(_):
    worker_info = get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    worker_id = worker_info.id
    batch_size = dataset.batch_size
    num_workers = worker_info.num_workers

    overall_start = 0
    overall_end = len(dataset)

    batches_per_worker = (overall_end - overall_start) // batch_size // num_workers
    examples_per_worker = batches_per_worker * batch_size

    worker_start = overall_start + worker_id * examples_per_worker
    if worker_id != num_workers - 1:
        # the first `num_workers - 1` workers load examples divisible by `batch_size`
        worker_end = min(worker_start + examples_per_worker, overall_end)
    else:
        # the last worker loads all remaining examples
        worker_end = overall_end

    # configure the dataset to only process the split workload

    """
    use `worker_start` and `worker_end` to configure current worker
    """