import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def __init__(self):
        self.normal = torch.distributions.normal.Normal(0, 1)
        super().__init__()

    def forward(self, x):
        return self.normal.sample(x.shape)


model = Model()
x = torch.randn(2, 3)
print(model(x))
print(torch.onnx.dynamo_export(model, x).model_proto)

def eval_function(  # type: ignore[override]
        self,
        function: onnxscript.OnnxFunction,
        args: Sequence[ValidArgumentType],
        kwargs: Mapping[str, ValidArgumentType],
    ):
        # args/kwargs are TorchScriptTensor/python built-in based
        param_schemas = function.param_schemas()
        (
            inputs,
            attributes,  # <============ ?
        ) = param_manipulation.separate_input_attributes_from_arguments(
            param_schemas, args, kwargs, fill_defaults=True, allow_extra_kwargs=True
        )