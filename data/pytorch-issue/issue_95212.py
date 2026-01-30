import torch
import torch.nn as nn

batch_inputs = torch.randn(2, 3, 16, 16)
bn_train = torch.nn.BatchNorm2d(3).train()
bn_eval = torch.nn.BatchNorm2d(3).eval()
batch_outputs_train = bn_train(batch_inputs)
single_output_train = bn_train(batch_inputs[0].unsqueeze(0))
batch_outputs_eval = bn_eval(batch_inputs)
single_output_eval = bn_eval(batch_inputs[0].unsqueeze(0))
print((batch_outputs_train[0] == single_output_train).all())  # False
print((batch_outputs_eval[0] == single_output_eval).all())   # True