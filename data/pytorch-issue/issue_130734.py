import torch
import torch.nn as nn

def test_saved_tensors_hooks_gc_segfault():
    shape = (101, 103)
    for i in range(10):
        print("**** iter", i)
        v = torch.nn.Parameter(torch.randn(shape))

        class _Handler:
            def __init__(self):
                self.scope = torch.autograd.graph.saved_tensors_hooks(self._pack_hook, self._unpack_hook)
                self.scope.__enter__()
                self.exited = False

            def _pack_hook(self, x):
                print(f"*** _pack_hook {self}")
                return x

            def _unpack_hook(self, x):
                print(f"*** _unpack_hook {self}")
                if not self.exited:
                    self.exited = True
                    print(f"*** exit {self.scope}, pack_hook {hex(id(self.scope.pack_hook))}")
                    self.scope.__exit__()
                return x

        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            handler = _Handler()  # keep ref...  # noqa
            x = v * torch.randn(shape)
            x.sum().backward()

test_saved_tensors_hooks_gc_segfault()