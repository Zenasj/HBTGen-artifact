import torch

model = capture_pre_autograd_graph(model)
model = prepare_qat_pt2e(model, ...)
train(model)
model = convert_pt2e(model)  # dropout subgraph rewriting happens here
inference(model)

model = capture_pre_autograd_graph(model)
model = prepare_qat_pt2e(model, ...)
train(model)
model = convert_pt2e(model)
torch.ao.quantization.move_model_to_eval(model)  # dropout subgraph rewriting happens here
inference(model)