import torch.nn as nn

import torch

# define NN architecture
class PredictLiquidationsV1(torch.nn.Module):
    def __init__(self, input_features, output_features, hidden_units):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.linear1 = torch.nn.Linear(in_features=input_features, out_features=hidden_units)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p = 0.2)
        self.linear2 = torch.nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(p = 0.4)
        self.linear3 = torch.nn.Linear(in_features=hidden_units, out_features=output_features)
        self.dequant = torch.ao.quantization.DeQuantStub()        
        
    def forward(self, x):
        x = self.quant(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = self.dequant(x)
        return x
    
    def fuse_model(self):
        torch.ao.quantization.fuse_modules(self, [['linear1', 'relu1']], inplace=True)
        torch.ao.quantization.fuse_modules(self, [['linear2', 'relu2']], inplace=True)
    
# instantiate the model
model_1 = PredictLiquidationsV1(input_features=41, output_features=1, hidden_units=82)

model_1.fuse_model()

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.001, weight_decay=0.01)

model_1.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
torch.ao.quantization.prepare_qat(model_1, inplace=True)

# Train model
# ...

quantized_model_1 = torch.ao.quantization.convert(model_1.eval(), inplace=False)
quantized_model_1.eval()
# Export the quantized_model_1 (requires onnx to be installed in the env)
torch.onnx.export(quantized_model_1,
                  torch.randn((1,41), requires_grad = True),
                  'quantized_model_1.onnx',
                  opset_version=16,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})