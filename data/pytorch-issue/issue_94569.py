import torch

def propagate(self, *args):
        with self._mode:
            fake_args = [self._mode.from_tensor(a) for a in args]
            return super().run(*fake_args)

def propagate(self, *args):
        with self._mode:
            fake_args = [self._mode.from_tensor(a) if isinstance(a, torch.Tensor) else None for a in args]
            return super().run(*fake_args)