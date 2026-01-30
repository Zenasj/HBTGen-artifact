import torch

from typing import Annotated, Iterator

from torch.utils.data import IterableDataset


class MyIterableDataset(IterableDataset[Annotated[int, "foo"]]):
    def __iter__(self) -> Iterator[Annotated[int, "foo"]]:
        yield 123


ds = MyIterableDataset()