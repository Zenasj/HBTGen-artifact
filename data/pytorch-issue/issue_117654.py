import torch

def test_index_end(self):
        def f(x, y):
            i = y.item()
            torch._check(x.size(0) > i)
            return x[i]

        gm = make_fx(f, tracing_mode="symbolic")(torch.randn(16), torch.tensor(15))
        self.assertExpectedInline(show_guards(gm), """""")