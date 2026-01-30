import torch

python
@torch.jit.script
class MyCell(object):
    @staticmethod
    def do_it(x, h):
        new_h = torch.tanh(x + h)
        return new_h, new_h