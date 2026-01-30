import torch

def test_export_tensor_bool_not(self):
        def true_fn(x, y):
            return x + y

        def false_fn(x, y):
            return x - y

        def f(x, y):
            return cond(not torch.any(x), true_fn, false_fn, [x, y])