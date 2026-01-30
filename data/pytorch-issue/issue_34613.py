import torch

model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model.fuse_model()
model_backbone_prep = torch.quantization.prepare(model)
sample_input = torch.rand(1, 3, 224, 224, device='cpu')
model_backbone_prep(sample_input)
q_model = torch.quantization.convert(model_backbone_prep)
q_model(sample_input)