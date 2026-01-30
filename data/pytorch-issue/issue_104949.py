import torch
import torch.nn as nn

class MyModel(nn.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(10,  1)

    def forward(self, input_a, input_b=None):    # the input_b is not actually used in the forward computation 
         return self.layer(input_a)

#  Here we export the trained model!  the file "MyModel_checkpoint_file.pt" is the training result
my_model = torch.load("MyModel_checkpoint_file.pt")
my_model.eval()

input_names = ["input_a", "input_b"]
output_names = ["output"]
fake_input_a = torch.rand(10)
fake_input_b = torch.rand(10)
with torch.no_grad():
    torch.onnx.export(
         model=my_model,
         args = (fake_input_a, fake_input_b),
         f = "result.onnx",
         input_names = input_names,
         output_names = output_names
    )

import onnxruntime
session = onnxruntime.InferenceSession("result.onnx", providers=["CPUExecutionProvider"])
for input in session.get_inputs():
    print(input.name)