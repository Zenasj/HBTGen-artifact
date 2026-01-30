import torch

image = get_dummy_input()
image_batch = image.repeat(128, 1, 1, 1)

onnx_model = torch.onnx.export(
    lightning_model,
    image_batch,
    input_names=['input'],
    output_names=['output'],
    dynamo=True,
    dynamic_shapes=[{0: torch.export.Dim('batch_size', min=1, max=128)}],
    optimize=True,
    verbose=True  # Includes metadata used during quantization
)

onnx_model.save(output_path)