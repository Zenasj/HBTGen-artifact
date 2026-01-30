import torch
import torch.nn as nn

for mode in range(1, 6):
            _variable = 0
            _variable_2 = 0

            mod = MyModule(mode=mode)
            model = torch._dynamo.optimize(backend="eager", nopython=mode != 6)(mod)
            assert _variable == 0
            assert _variable_2 == 0

            model(torch.tensor([1]))
            assert _variable == 1
            assert _variable_2 == 0

            model(torch.tensor([1]))
            assert _variable == 2
            assert _variable_2 == 0

class MyModule(torch.nn.Module):
    def __init__(self, mode: int):
        super().__init__()
        self.mode = mode
        self.register_forward_pre_hook(self.pre_forward, with_kwargs=True)

    def pre_forward(self, module, args, kwargs):
        if self.mode == 5:
            if user_function():
                global _variable
                _variable += 1
        return args, kwargs

    def forward(self, x):
        global _variable, _variable_2

        if self.mode == 1:
            if torch._utils.is_compiling():
                _variable += 1
            else:
                _variable_2 += 1
        elif self.mode == 2:
            if user_function():
                _variable += 1
        elif self.mode == 3:
            lambda_f = lambda: torch._utils.is_compiling()  # noqa: E731
            if lambda_f():
                _variable += 1
        elif self.mode == 4:
            for cond in user_generator():
                if cond:
                    _variable += 1
        elif self.mode == 5:
            x += 1
        elif self.mode == 6:
            if user_function():
                torch._dynamo.graph_break()
                _variable += 1
        return x