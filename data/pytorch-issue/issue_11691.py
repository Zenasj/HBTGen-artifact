import numpy as np
import random

dataset = SpeechAnimationDataset(opt)

loader = DataLoader(dataset, batch_size=5, num_workers=2) #works
loader = DataLoader(dataset, batch_size=5, num_workers=2, shuffle=True) #fails after long process
data_iter = iter(loader)
for i in tqdm(range(500)):
    data = next(data_iter)

class RandomSampler(Sampler):
    r"""Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        arr = np.arange(len(self.data_source))
        np.random.shuffle(arr)
        return iter(arr)

    def __len__(self):
        return len(self.data_source)