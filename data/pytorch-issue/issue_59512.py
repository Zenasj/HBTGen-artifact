import torch
import numpy as np

from torch.utils.data import TensorDataset, Subset
dataset = TensorDataset(torch.tensor([1, 2, 3]))
subset_of_dataset = Subset(dataset , list(range(3)))
print(subset_of_dataset[:])

(tensor([1, 2, 3]),)

from torch.utils.data import TensorDataset, Subset
dataset = TensorDataset(torch.tensor([1, 2, 3]))
subset_of_dataset = Subset(dataset , list(range(3)))
subset_of_subset =  Subset(subset_of_dataset , list(range(3)))
print(subset_of_subset[:])

from torch.utils.data import TensorDataset, Subset
dataset = TensorDataset(torch.tensor([1, 2, 3]))
subset_of_dataset = Subset(dataset , list(range(3)))
subset_of_subset =  Subset(subset_of_dataset , list(range(3)))
print(subset_of_subset[0])

(tensor(1),)

py
train, test = random_split(ds, [100, 10])
train, val = random_split(train, [90, 10])

dataset = TensorDataset(torch.tensor([1, 2, 3]))
train, test = random_split(dataset ,  [2,1])
print(train[:])    # Can be sliced
train, val = random_split(train, [1, 1])
print(train[:])    # Can not be sliced

label_frequency = np.bincount(train[:][-1])
weights, num_sample = my_weight_generator(label_frequency)
train_dataloader = DataLoader(train, sampler=WeightedRandomSampler(weights, num_sample, replacement=True), batch_size=32)

d = TensorDataset(torch.tensor([1, 2, 3]), torch.tensor([1, 1, 0]))
s_of_d = Subset(d, list(range(3)))
s_of_s =  Subset(s_of_d , list(range(3)))

s_of_d.indices = torch.LongTensor(s_of_d.indices)
label_frequency = np.bincount(s_of_s[:][-1])

label_frequency = np.bincount([i[-1] for i in s_of_s])

label_frequency = np.bincount(d[s_of_s.indices][-1])

label_frequency = np.bincount(s_of_d[:][-1])

py
dataset = TensorDataset(torch.tensor([1, 2, 3]))
train, test = random_split(dataset ,  [2,1])
print(train[:])    # Can be sliced