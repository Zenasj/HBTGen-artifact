import torch

model_inputs = model_inputs.cpu()
model = model.cpu()

# traced_module = torch.jit.script(model, model_inputs)
# print(traced_module.code)

torch.onnx.export(
        model,
        model_inputs,
        output_,
        verbose=True,
        input_names=input_names,
        output_names=output_names)