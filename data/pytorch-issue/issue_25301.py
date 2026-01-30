import torch.nn as nn

import os
from torch import nn
from torch.multiprocessing import Queue, Process, set_start_method
from time import sleep
from copy import deepcopy

def actor(queue):
    print(f'Actor started on PID #{os.getpid()}')
    while True:
        if not queue.empty():
            queue.get()
            print('Actor stepped')
        sleep(.01)
    
def learner(queue):
    print(f'Learner started on PID #{os.getpid()}')
    net = nn.LSTM(1, 1).cuda()
    while True:
        if not queue.full():
            queue.put(deepcopy(net.state_dict()))
            print('Learner stepped')
        sleep(.01)

def run():
    queue = Queue(1)

    processes = dict(
        a=Process(target=actor, args=(queue,)),
        l=Process(target=learner, args=(queue,)))

    for p in processes.values():
        p.start()

    for p in processes.values():
        p.join()

if __name__ == '__main__':
    set_start_method('spawn')
    run()

import torch
from ctypes import c_int, c_bool, Structure
import torch.multiprocessing as mp
import aljpy

log = aljpy.logger()
torch.set_num_threads(1)

### DOUBLE BUFFER ###

def move(x, y):
    if isinstance(x, dict):
        for k in x:
            move(x[k], y[k])
    else:
        y.copy_(x)

def copy(x):
    if isinstance(x, dict):
        return x.__class__({k: copy(v) for k, v in x.items()})
    else:
        return x.clone()

class DoubleBufferAccess(Structure):
    _fields_ = [('writing', c_int), ('reading', c_int), ('active', c_int), ('swap', c_bool)]

class DoubleBufferWriter:

    def __init__(self, dbuf, buffers):
        self._access = dbuf._access
        self._buffers = buffers 
        dbuf._wconn.send(buffers)
        dbuf._wconn.close()

    def __call__(self, data):
        a = self._access

        with a.get_lock():
            a.writing += 1
            active = a.active

        move(data, self._buffers[active])

        with a.get_lock():
            if (a.reading + a.writing) == 1:
                a.active = 1-active
                a.swap = False
            else:
                a.swap = True

            a.writing -= 1

class DoubleBufferReader:

    def __init__(self, dbuf):
        self._access = dbuf._access
        self._buffers = dbuf._rconn.recv()
        dbuf._rconn.close()

    def __call__(self):
        a = self._access

        with a.get_lock():
            a.reading += 1
            active = a.active
        
        data = copy(self._buffers[1-active])

        with a.get_lock():
            if a.swap and ((a.reading + a.writing) == 1):
                a.active = 1-active
                a.swap = False
            a.reading -= 1
        
        return data

class DoubleBuffer:
    """Roughly follows https://stackoverflow.com/questions/23666069/single-producer-single-consumer-data-structure-with-double-buffer-in-c
    """

    def __init__(self):
        self._rconn, self._wconn = mp.Pipe(duplex=False)
        self._access = mp.Value(DoubleBufferAccess, 0, 0, 0, lock=True)

    def writer(self, first, second):
        return DoubleBufferWriter(self, (first, second))

    def reader(self):
        return DoubleBufferReader(self)

### TESTS ###

COUNT = 500000

def writer(dbuf):
    print('Writer started')
    writer = dbuf.writer(torch.tensor((-1,)), torch.tensor((-1,)))
    for i in torch.arange(COUNT+1):
        writer(i)
    print('Writer stopped')

def reader(dbuf):
    print('Reader started')
    reader = dbuf.reader()
    prev, count, first = -1, 0, -1
    while True:
        data = reader()
        assert data >= prev, (data, prev)
        prev = data.clone()

        count += 1
        if count == 1:
            first = prev

        if data == COUNT:
            break
    print(f'Saw {count} values; {count/(COUNT - float(first)):.0%} of all values')

def test():
    dbuf = DoubleBuffer()

    w = mp.Process(target=writer, args=(dbuf,), daemon=True)
    r = mp.Process(target=reader, args=(dbuf,), daemon=True)

    w.start()
    r.start()

    w.join()
    r.join()

if __name__ == '__main__':
    test()