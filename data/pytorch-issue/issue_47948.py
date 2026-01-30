import torch

def test_assertEqual(self, device):
        a = make_tensor((), device, torch.int32, low=-5, high=5)
        self.assertEqual(a.cpu().numpy(), a, rtol=0, atol=0)