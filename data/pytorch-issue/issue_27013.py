import torch

def test_fmod_error(self):
        t1 = torch.rand((3, 3))
        t2 = torch.rand(1)
        dist.rpc('worker{}'.format(self.rank + 1), torch.fmod,
            args=(t1, t2))