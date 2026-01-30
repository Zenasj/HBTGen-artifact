numerator = torch.abs(model_outputs - target_outputs)
denominator = torch.abs(model_outputs) + torch.abs(target_outputs)
elementwise_smape = torch.div(numerator, denominator)
nan_mask = torch.isnan(elementwise_smape)
loss = elementwise_smape[~nan_mask].mean()
assert ~torch.isnan(loss)  # loss = 0.023207199
loss.backward()

import torch


model_outputs = torch.load('model_outputs.pt')
target_outputs = torch.load('target_outputs.pt')
print(model_outputs)
print(target_outputs)

model_outputs.requires_grad = True
target_outputs.requires_grad = False

numerator = torch.abs(model_outputs - target_outputs)
denominator = torch.abs(model_outputs) + torch.abs(target_outputs)
elementwise_smape = torch.div(numerator, denominator)
nan_mask = torch.isnan(elementwise_smape)
loss = elementwise_smape[~nan_mask].mean()
assert ~torch.isnan(loss)
print(loss.detach().numpy())
loss.backward()
print('Success!')

nan_mask = torch.isnan(elementwise_smape)
loss = model_outputs[~nan_mask].mean()
loss.backward()  # succeeds

nan_mask = torch.isnan(elementwise_smape)
loss = numerator[~nan_mask].mean()
loss.backward()  # succeeds

nan_mask = torch.isnan(elementwise_smape)
loss = denominator[~nan_mask].mean()
loss.backward()  # succeeds