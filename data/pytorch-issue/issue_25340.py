import gc
import time
import torch
import weakref


class Test(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, refs):
        refs.append(weakref.ref(ctx))
        y = x.clone()
        ctx.y = y
        return y


if __name__ == '__main__':
    refs = []  # weakrefs to ctx objects
    x = torch.rand(8, 8, requires_grad=True)
    y = Test.apply(x, refs)
    print(refs)
    del y
    gc.collect()
    # Another bug mentions sleeping somehow helping. It doesn't
    # help here.
    time.sleep(20)
    # the weakref should died by now
    print(refs)  # Bug! Weakref is still alive