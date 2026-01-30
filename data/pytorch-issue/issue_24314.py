import torch
import torch.nn as nn

class Foo(torch.nn.Module):
    def __init__(self, inner_module, trace_inputs, some_option=1.0, some_other=None, **kwargs):
        self.inner = torch.jit.trace_inputs(inner_module, trace_inputs)
        self.some_config = some_option
        self.register_buffer('_my_fancy_buffer', self.method_that_uses_librosa_or_scipy_or_whatever())

    @torch.jit.export
    def inference(self, inputs):
        inner_outputs = self.inner(inputs)
        return self.do_some_postprocessing_using_fancy_buffer(inner_outputs)