import torch
import multiprocessing as mp


def foo(q):
    x = torch.Tensor([1.0]).cuda()
    print(x)
    q.put("Hello")


if __name__ == '__main__':
    q = mp.Queue()
    p = mp.Process(target=foo, args=(q,))
    p.start()
    print(q.get())
    p.join()

import multiprocessing as mp


def foo(q):
    import torch
    x = torch.Tensor([1.0]).cuda()
    print(x)
    q.put("Hello")


if __name__ == '__main__':
    q = mp.Queue()
    p = mp.Process(target=foo, args=(q,))
    p.start()
    print(q.get())
    p.join()