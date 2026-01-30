import torch.nn as nn
import random

import torch

import torch.multiprocessing as mp

from copy import deepcopy

from functools import partial

from time import *

from torchvision import models

import numpy as np

from tqdm import tqdm

def parallel_produce(

    queue: mp.Queue,

    model_method,

    i

) -> None:

    pure_model: torch.nn.Module = model_method()

    # if you delete this line, model can be passed
    pure_model.to('cuda')

    pure_model.share_memory()

    while True:

        corrupt_model = deepcopy(pure_model)

        dic = corrupt_model.state_dict()

        dic[list(dic.keys())[0]]*=2

        corrupt_model.share_memory()

        queue.put(corrupt_model)

def parallel(

    valid,

    iteration: int = 1000,

    process_size: int=2,

    buffer_size: int=2

):

    pool = mp.Pool(process_size)

    manager = mp.Manager()

    queue = manager.Queue(buffer_size)

    SeedSequence = np.random.SeedSequence()

    model_method = partial(models.squeezenet1_1,True)

    async_result = pool.map_async(

        partial(

            parallel_produce,

            queue,

            model_method,

        ),

        SeedSequence.spawn(process_size),

    )

    time = 0

    for iter_times in tqdm(range(iteration)):

        start = monotonic_ns()

        # this takes a long time

        corrupt_model: torch.nn.Module = queue.get()

        time += monotonic_ns() - start

        corrupt_model.to("cuda")

        corrupt_result = corrupt_model(valid)

        del corrupt_model

    pool.terminate()

    print(time / 1e9)

if __name__ == "__main__":

    valid = torch.randn(1,3,224,224).to('cuda')

    parallel(valid)

import torch

import torch.multiprocessing as mp

from copy import deepcopy

from functools import partial

from time import *

from torchvision import models

import numpy as np

from tqdm import tqdm

def parallel_produce(

    queue: mp.Queue,

    model_method,

    iterations,

    i,

 
) -> None:
    pure_model: torch.nn.Module = model_method()
    # if you delete this line, model can be passed
    pure_model.to('cuda')

    for j in range(iterations):
        corrupt_model = deepcopy(pure_model)
        dic = corrupt_model.state_dict()
        dic[list(dic.keys())[0]]*=2
        queue.put(corrupt_model.state_dict())

def parallel(

    valid,

    iteration: int = 1000,

    process_size: int=2,

    buffer_size: int=2

):
    
    model_method = partial(models.squeezenet1_1,True)
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(process_size)
    queue = ctx.Queue(buffer_size)

    proc = []
    for i in range(process_size):
        p  = ctx.Process(target = parallel_produce, args=(queue,model_method,iteration // process_size,i,))
        p.start()
        proc.append(p)

    time = 0
    
    corrupt_model = model_method()

    for iter_times in tqdm(range(iteration)):        
        start = monotonic_ns()
        # this takes a long time
        state = queue.get()
        corrupt_model.load_state_dict(state)
        time += monotonic_ns() - start
        corrupt_model.to("cuda")
        corrupt_result = corrupt_model(valid)

    for p in proc:
        p.join()

    print(time / 1e9)

if __name__ == "__main__":

    valid = torch.randn(1,3,224,224).to('cuda')

    parallel(valid)

import torch

import torch.multiprocessing as mp

from copy import deepcopy

from functools import partial

from time import *

from torchvision import models

import numpy as np

from tqdm import tqdm

def parallel_produce(

    queue: mp.Queue,

    model_method,

    iterations,

    i,

 
) -> None:
    pure_model: torch.nn.Module = model_method()
    # if you delete this line, model can be passed
    pure_model.to('cuda')

    for j in range(iterations):
        corrupt_model = deepcopy(pure_model)
        dic = corrupt_model.state_dict()
        dic[list(dic.keys())[0]]*=2
        queue.put(corrupt_model.state_dict())

def parallel(

    valid,

    iteration: int = 1000,

    process_size: int=2,

    buffer_size: int=2

):
    
    model_method = partial(models.squeezenet1_1,True)
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(process_size)
    queue = ctx.Queue(buffer_size)

    proc = []
    for i in range(process_size):
        p  = ctx.Process(target = parallel_produce, args=(queue,model_method,iteration // process_size,i,))
        p.start()
        proc.append(p)

    time = 0
    
    corrupt_model = model_method()
    sleep(3)
    for iter_times in tqdm(range(iteration)):        
        start = monotonic_ns()
        # this takes a long time
        state = queue.get(block = False)
        corrupt_model.load_state_dict(state)
        time += monotonic_ns() - start
        corrupt_model.to("cuda")
        corrupt_result = corrupt_model(valid)

    for p in proc:
        p.join()

    print(time / 1e9)

if __name__ == "__main__":

    valid = torch.randn(1,3,224,224).to('cuda')

    parallel(valid)