from typing import Dict
import io
import torch
import torch.nn as nn

class SampleModule(nn.Module):
    def __init__(self):
        super(SampleModule, self).__init__()
        self.a = 64

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        data = input_dict["input"]
        x = torch.arange(self.a, device=data.device)
        y = x[None, None, :, None]
        z = torch.ones_like(data[:, :1], device=data.device) * y
        return z

    def gen_inputs(self):
        return {"input":torch.rand((1,2,64,64), device=torch.device('cuda:0'))}

def test_reproducer():
    model = SampleModule()
    model.cuda()
    model.eval()

    inputs = model.gen_inputs()
    input_names = ["input"]
    output_names = ["output"]

    onnx_bytes = io.BytesIO()
    #dynamic_axes = {"input": {0: "batch_size"}}
    empty_dynamic_axes = {}

    print("before onnx export")
    torch.onnx.export(
        model=model,
        args=inputs,
        f=onnx_bytes,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=empty_dynamic_axes,
        opset_version=12,
        export_params=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    )
    print("after onnx export")