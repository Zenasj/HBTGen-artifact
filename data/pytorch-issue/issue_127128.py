import torch

dummy_input = torch.randn(
    1, 100, 400
).to(device)
dummy_layer_one_mask = torch.randint(
    0, 2, (1, 10)
).to(device)
dummy_layer_two_mask = torch.randint(
    0, 2, (1,10)
).to(device)

onnx_model = torch.onnx.dynamo_export(
    actor_model,
    (dummy_input, dummy_layer_one_mask, dummy_layer_two_mask),
)

onnx_model.save(f"{model_directory}/model/actor.onnx")