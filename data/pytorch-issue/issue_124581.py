import torch

def test_reshape_divisibility_unbacked(self):
        def f(x):
            i0 = x.item()
            r = torch.zeros(i0, 4, 20)
            r = r.transpose(2, 1)
            return r.reshape(-1, 80)
        make_fx(f, tracing_mode="symbolic")(torch.tensor(24))