import torch.nn as nn

import io
import onnx
import torch
import onnxruntime

ONNX_OPSET = 12

def export(
    model,
    dummy_inputs,
    input_names,
    output_names,
    input_dynamic_axes,
    output_dynamic_axes,
):
    with io.BytesIO() as onnx_bytes:
        torch.onnx.export(
            model,
            dummy_inputs,
            onnx_bytes,
            export_params=True,
            do_constant_folding=True,
            opset_version=ONNX_OPSET,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={**input_dynamic_axes, **output_dynamic_axes},
        )

        return onnx_bytes.getvalue()


model = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=8)

input = torch.arange(10)
offsets = torch.tensor([0, 2, 4, 4, 6])

print(model(input, offsets))

dummy_inputs = input, offsets
input_names = ["input", "offsets"]
output_names = ["output"]
input_dynamic_axes = {"input": {0: "N"}, "offsets": {0: "B"}}
output_dynamic_axes = {"output": {0: "B"}}

model_proto = export(
    model,
    dummy_inputs,
    input_names,
    output_names,
    input_dynamic_axes,
    output_dynamic_axes,
)

session = onnxruntime.InferenceSession(model_proto)
print(session.run(None, {"input": input.numpy(), "offsets": offsets.numpy()}))