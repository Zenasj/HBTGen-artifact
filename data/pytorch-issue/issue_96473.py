import torch.nn as nn

import torch
import torch._dynamo
import logging
from functorch.experimental.control_flow import cond
import torch.quantization
from torch.ao.quantization.quantize_fx import prepare_qat_fx
from torch.ao.quantization.experimental.qconfig import uniform_qconfig_8bit

class MyModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        qconfig_dict = {"object_type": [(torch.nn.Linear, uniform_qconfig_8bit)]}
        example_inputs = (torch.randn(5, 5),)
        self.model = torch.nn.Linear(5, 5)
        self.quantized_model = prepare_qat_fx(self.model, qconfig_dict, example_inputs=example_inputs)

    def forward(self, pred, x):
        def true_fn(x):
            return x.sin() + self.quantized_model(x)

        def false_fn(x):
            return x.cos() + self.model(x)

        return cond(pred, true_fn, false_fn, [x])

module = MyModule()
pred = torch.tensor(True)
x = torch.rand((5, 5))
opt_m = torch._dynamo.optimize("eager")(module)
print(module(pred, x))
print(opt_m(pred, x))

cond/map

tracked_fakes

def SHAPE_ENV(self, guard: Guard):
        # Let's handle ShapeEnv guards.  To do this, we will resolve
        # shape variables to sources from tracked_fakes.  This must happen after
        # tensor checks.
        assert guard.name == ""
        output_graph = self.check_fn_manager.output_graph
        # NB: self.output_graph can be None in the debug_nops tests
        fs = output_graph.tracked_fakes
        guards = output_graph.shape_env.produce_guards(
            [a.fake for a in fs],
            [a.source for a in fs],
            source_ref=self.source_ref,
        )
        for shape_guard in guards:
            self._produce_guard_code(guard, [shape_guard], shape_env=True)