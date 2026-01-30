import torch
import math
import time

class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    def give_data(self, start, end):
        for i in range(start, end):
            if i > 10:
                time.sleep(2)
            yield i

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return self.give_data(iter_start, iter_end)
    
if __name__ == "__main__":
    ds = MyIterableDataset(start=0, end=20)

    # Mult-process loading with two worker processes
    for item in (torch.utils.data.DataLoader(ds, num_workers=2, batch_size=2)):
        print(item)

tensor([0, 1]) # Loaded fast
tensor([10, 11])  # Loaded slowly
tensor([2, 3])  # Loaded fast
tensor([12, 13])  # Loaded slowly
tensor([4, 5])  # Loaded fast
tensor([14, 15])  # Loaded slowly
tensor([6, 7])  # Loaded fast
tensor([16, 17])  # Loaded slowly
tensor([8, 9])  # Loaded fast
tensor([18, 19])  # Loaded slowly

tensor([0, 1]) # Loaded fast
tensor([2, 3])  # Loaded fast
tensor([4, 5])  # Loaded fast
tensor([6, 7])  # Loaded fast
tensor([10, 11])  # Loaded slowly
tensor([8, 9])  # Loaded fast
tensor([12, 13])  # Loaded slowly
tensor([14, 15])  # Loaded slowly
tensor([16, 17])  # Loaded slowly
tensor([18, 19])  # Loaded slowly