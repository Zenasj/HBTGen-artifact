import torch.nn as nn

import torch
import multiprocessing
import atexit

def split_loader_creator():
    for i in range(20):
        yield torch.zeros(10, 170, 70)

def background_generator_helper(gen_creator):
    def _bg_gen(gen_creator, conn):
        gen = gen_creator()
        while conn.recv():
            try:
                conn.send(next(gen))
            except StopIteration:
                conn.send(StopIteration)
                return
            except Exception:
                import traceback
                traceback.print_exc()

    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=_bg_gen, args=(gen_creator, child_conn))
    p.start()
    atexit.register(p.terminate)

    parent_conn.send(True)
    while True:
        parent_conn.send(True)
        x = parent_conn.recv()
        if x is StopIteration:
            return
        else:
            yield x

def background_generator(gen_creator): # get several processes in the background fetching batches in parallel to keep up with gpu
    generator = background_generator_helper(gen_creator)
    while True:
        batch = next(generator)
        if batch is StopIteration:
            return
        yield batch

torch.zeros(152*4, 168*4).float()

data_loader = background_generator(split_loader_creator)
for i, batch in enumerate(data_loader):
    print(i)