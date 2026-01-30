import logging
import torch

logging.basicConfig(format='[%(asctime)s|%(filename)s:%(lineno)d] %(message)s', level=logging.DEBUG)

@torch.enable_grad()
def bar(depth):
    if depth>0:
        bar(depth-1)
    return 42

def foo(depth):
    logging.info('grad enabled: {}'.format(torch.is_grad_enabled()))
    x = bar(depth)
    logging.info('grad enabled: {}'.format(torch.is_grad_enabled()))
    return x

with torch.no_grad():
    assert not torch.is_grad_enabled()
    y = foo(0)
    assert not torch.is_grad_enabled()

    y = foo(1)
    assert not torch.is_grad_enabled() # FAILS