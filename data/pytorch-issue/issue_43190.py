import torch
from torch.futures import Future
from torch.testing._internal.common_utils import TemporaryFileName
fut = Future[int]()
with TemporaryFileName() as fname:
    torch.save(fut, fname)