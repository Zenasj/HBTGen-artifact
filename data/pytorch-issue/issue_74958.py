import torch

py
import pickle

from torch.utils.data.graph import traverse
from torchdata.datapipes.iter import IterableWrapper


def foo(bar):
    def wrapper(data):
        return data, bar

    return wrapper


dp = IterableWrapper("abc").map(foo("bar"))

traverse(dp, only_datapipe=False)  # passes
traverse(dp, only_datapipe=True)  # passes
pickle.dumps(dp)  # fails

py
import pickle

from torchdata.datapipes.iter import IterDataPipe, IterableWrapper, Zipper


class CustomIterDataPipe(IterDataPipe):
    def classify(self, x):
        return 0

    def __init__(self):
        self._dp = Zipper(*IterableWrapper([]).demux(2, self.classify))

    def __iter__(self):
        yield from self._dp


pickle.dumps(CustomIterDataPipe())

py
dataset = ...

# Maybe we can make this is a static method of the DataLoader2?
try:
    num_workers = len(os.sched_getaffinity(0))
except Exception:
    num_workers = os.cpu_count() or 1

for parallelism_mode in ["mp", "thread"]:
    dp = dataset.batch(2, drop_last=parallelism_mode == "thread").collate()

    dl = DataLoader2(
        dp,
        batch_size=None,
        shuffle=True,
        num_workers=num_workers,
        parallelism_mode=parallelism_mode,
        timeout=1,
    )

    for _ in dl:
        pass