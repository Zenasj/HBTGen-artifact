import torch

def test_fake_crossref_backward_amp(self, device, dtype, op):
        self._test_fake_crossref_helper(device, dtype, op, torch.cuda.amp.autocast)