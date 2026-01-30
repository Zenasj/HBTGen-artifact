import torch

# create a model instance
model_fp32 = M()

# model must be set to eval mode for fusion to work
model_fp32.eval()

# model must be set to train mode for QAT logic to work
model_fp32.train()

# Prepare the model for QAT. This inserts observers and fake_quants in
# the model that will observe weight and activation tensors during calibration.
model_fp32_prepared = torch.quantization.prepare_qat(model_fp32_fused)

# model must be set to train mode for QAT logic to work
model_fp32_prepared.train()