import onnx
import torch

model = torch.load('save_entire_model')
text_input = torch.randint(low=0, high=5000, size=(10, 10))    
input_names = ['inputs']
output_names = ['predictions']
dummy_inputs = (text_input, text_input, text_input, text_input)
torch.onnx.export(model, args=dummy_inputs, input_names=input_names,
                      output_names=output_names, f='some_file')

def forward(self, images_a, images_b):

        raise NotImplementedError