import gc
import torch
from torch.nn import Linear
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR


def main():
    t = torch.randn(100, 100)
    model = Linear(100, 100)

    optimizer = Adam(params=model.parameters())
    scheduler = MultiStepLR(optimizer=optimizer, milestones=[1, 2, 3])
    t = model(t)
    t.sum().backward()
    optimizer.step()
    scheduler.step()


if __name__ == '__main__':
    gc.enable()
    gc.set_debug(gc.DEBUG_SAVEALL)
    main()
    garbage = gc.collect()

    try:
        assert not garbage, "Memory leak is occurred!"
    except AssertionError:
        import objgraph
        objgraph.show_backrefs(gc.garbage, filename='finalizers.png')
        raise