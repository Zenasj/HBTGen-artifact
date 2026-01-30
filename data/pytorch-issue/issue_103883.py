import torch.nn as nn

from typing import Tuple, Union
import torch


class CustomClassOp_Func(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(
        g: torch.onnx._internal.jit_utils.GraphContext,
        result: Union[torch.Value, Tuple[torch.Value]],
        *args: torch.Value,
    ) -> torch.Value:
        return result

    @staticmethod
    def symbolic(
        g: torch.onnx._internal.jit_utils.GraphContext,
        result: Union[torch.Value, Tuple[torch.Value]],
        *args: torch.Value,
    ) -> torch.Value:
        return g.op(
            f"CustomDomain::custom_op",
            *args,
            outputs=len(result),
        )


class Custom(torch.nn.Module):
    def real_compute(self, input1, input2):
        a = input1 + input2
        b = 2 * input1
        c = 2 * input2
        return a,b,c
    
    def fake_compute(self, input1, input2):
        a = input1
        b = input1
        c = input2
        return a,b,c
    
    # Change me
    compute_to_use = real_compute # This doesn't work
#    compute_to_use = fake_compute # This work
    
    @torch.no_grad()
    def custom_op_inference(self, *args):
        return CustomClassOp_Func.apply(
                    self.compute_to_use(*args),
                    *args
                )
    
    def forward(self, input1, input2):
        if torch.onnx.is_in_onnx_export():
            args = input1, input2
            return self.custom_op_inference(*args)
        return self.compute_to_use(input1, input2)

model = Custom()
batch = (torch.FloatTensor(1, 3), torch.FloatTensor(1, 3))
torch.onnx.export(model, batch, "/tmp/model.onnx", opset_version=16, custom_opsets={"CustomDomain": 1})