import numpy as np
import torch
from torch.utils import data
import torch.utils.data as utils
import cProfile, pstats

bs = 2048

n_variables = np.loadtxt("../example_data/example1.txt_train", dtype='str').shape[1]-1
variables = np.loadtxt("../example_data/example1.txt_train", dtype = np.float32, usecols=(0,))

#epochs = 200*n_variables
epochs = n_variables
print(epochs)

for j in range(1,n_variables):
    v = np.loadtxt("../example_data/example1.txt_train", dtype = np.float32, usecols=(j,))
    variables = np.column_stack((variables,v))

f_dependent = np.loadtxt("../example_data/example1.txt_train", dtype = np.float32, usecols=(n_variables,))
f_dependent = np.reshape(f_dependent,(len(f_dependent),1))

factors = torch.from_numpy(variables)
factors = factors.to('mps')
factors = factors.float()

product = torch.from_numpy(f_dependent)
product = product.to('mps')
product = product.float()

my_dataset = utils.TensorDataset(factors,product) # create your dataset
my_dataloader = utils.DataLoader(my_dataset, batch_size=bs, shuffle=False) # create your dataloader

profiler = cProfile.Profile()
profiler.enable()

for epoch in range(epochs):
    print(epoch)
    for i, data in enumerate(my_dataloader):
        print(i)
        fct = data[0].float().to('mps')
        prd = data[1].float().to('mps')
        
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()

import numpy as np
import torch
from torch.utils import data
import torch.utils.data as utils
import cProfile, pstats

bs = 2048

n_variables = np.loadtxt("../example_data/example1.txt_train", dtype='str').shape[1]-1
variables = np.loadtxt("../example_data/example1.txt_train", dtype = np.float32, usecols=(0,))

#epochs = 200*n_variables
epochs = n_variables
print(epochs)

for j in range(1,n_variables):
    v = np.loadtxt("../example_data/example1.txt_train", dtype = np.float32, usecols=(j,))
    variables = np.column_stack((variables,v))

f_dependent = np.loadtxt("../example_data/example1.txt_train", dtype = np.float32, usecols=(n_variables,))
f_dependent = np.reshape(f_dependent,(len(f_dependent),1))

factors = torch.from_numpy(variables)
factors = factors.to('mps')
factors = factors.float()

product = torch.from_numpy(f_dependent)
product = product.to('mps')
product = product.float()

my_dataset = utils.TensorDataset(factors,product) # create your dataset
my_dataloader = utils.DataLoader(my_dataset, batch_size=bs, shuffle=False) # create your dataloader

profiler = cProfile.Profile()
profiler.enable()

for epoch in range(epochs):
    print(epoch)
    for i, data in enumerate(my_dataloader):
        #print(i)
        fct = data[0].float().to('mps')
        prd = data[1].float().to('mps')
        
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()