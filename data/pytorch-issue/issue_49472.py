py
import time
import torch.utils.data as torchdata
from tqdm import tqdm


class Dataset(torchdata.Dataset):
	def __init__(self, data):
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, item):
		return self.data[item]


class IterableDataset(torchdata.IterableDataset):
	def __init__(self, data):
		self.data = data

	def __len__(self):
		return len(self.data)

	def __iter__(self):
		for d in self.data:
			yield d


if __name__ == '__main__':
	x = list(range(100))
	dataset = torchdata.DataLoader(Dataset(x), batch_size=5)
	iterable_dataset = torchdata.DataLoader(IterableDataset(x), batch_size=5)

	for _d in tqdm(dataset, desc="Dataset"):
		time.sleep(.5)

	for _d in tqdm(iterable_dataset, desc="Iterable Dataset"):
		time.sleep(.5)