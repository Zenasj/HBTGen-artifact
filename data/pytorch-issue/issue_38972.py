import torch
import torch.nn as nn

def convert_to_tensor(module, input):
    torch_inputs = [x for x in input]
    for i, x in enumerate(input):
        if not isinstance(x, torch.Tensor):
            torch_inputs[i] = torch.tensor(x, requires_grad=True)
    return tuple(torch_inputs)


torch.nn.Module.register_global_forward_pre_hook(convert_to_tensor)
module = MyModule()
new_module = MyModule2()
input = numpy.ones((5,), dtype=numpy.float32)
module(input)
module2(input)

from chainer.link_hooks import TimerHook
hook = TimerHook()
with hook:
    trainer.run()
hook.print_report()