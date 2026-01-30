import torch

from torch import Tensor
from torch.distributions import Distribution
from torch.utils.data import Dataset

class ProceduralDataset(Dataset, ABC):
    @property
    @abstractmethod
    def distribution(self) -> Distribution:
        pass

    def __init__(self, num_samples: int):
        self._n = num_samples
        self._samples = None

    def __getitem__(self, i):
        if self._samples is None:
            self._samples = self.distribution.sample((self._n,))
        return self._samples[i], Tensor()

    def __len__(self):
        return self._n

    def __iter__(self):
        self._i = 0
        return self

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        self._i += 1
        return self[self._i - 1]