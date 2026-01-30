import torch.nn as nn

import torch

def oom():
    try:
        x = torch.randn(100, 10000, device=1)
        for i in range(100):
            l = torch.nn.Linear(10000, 10000)
            l.to(1)
            x = l(x)
    except RuntimeError as e:
        print(e)
        print('at iteration', i)

import torch

def oom(raise_=False):
    try:
        x = torch.randn(100, 10000, device=1)
        for i in range(100):
            l = torch.nn.Linear(10000, 10000)
            l.to(1)
            x = l(x)
    except RuntimeError as e:
        print(e)
        print('at iteration', i)
        if raise_:
            raise

oom(True)

try:
    oom(True)
except:
    oom()

from IPython.display import Javascript
display(Javascript('Jupyter.notebook.restart_run_all({"confirm":false})'))

tb = exc.__traceback__
while tb:
    try:
        tb.tb_frame.clear()
    except RuntimeError:
        pass
    else:
        # Using this code triggers that the ref actually goes out of scope, otherwise it does not!
        # https://github.com/python/cpython/issues/113939
        tb.tb_frame.f_locals  # noqa
    tb = tb.tb_next