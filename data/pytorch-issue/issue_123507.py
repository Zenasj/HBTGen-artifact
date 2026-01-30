import torch.nn as nn

import torch
from torch import nn
import torch.nn.quantized as nnq
from torch.quantization import get_default_qconfig, prepare, convert

# Define the model
class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(5, 10)  # Example dimensions

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = SimpleLinearModel()

# Define the qconfig (using 'fbgemm' or 'qnnpack' configuration)
qconfig = get_default_qconfig('fbgemm')  # or 'qnnpack'

# Apply the qconfig to the model
model.qconfig = qconfig

# Prepare the model for quantization
prepared_model = prepare(model, inplace=False)

# Convert the prepared model to a quantized model
quantized_model = convert(prepared_model, inplace=False)

# Now, quantized_model is ready for inference or further operations

torch.backends.quantized.engine = 'qnnpack'