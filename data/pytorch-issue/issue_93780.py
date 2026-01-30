import torch

def test_mutated_metadata(self):
        def model(x):
            y = x.view(0)
            x.resize_(20)
            x.fill_(2)
            y.fill_(3)

        from torchdynamo.optimizations.backends import aot_autograd
        with torchdynamo.optimize("aot_autograd"):
            for i in range(5):
                with self.subTest(i):
                    x = torch.empty(0)
                    r = model(x)
                    self.assertEqual(x, torch.full((20,), 2.))