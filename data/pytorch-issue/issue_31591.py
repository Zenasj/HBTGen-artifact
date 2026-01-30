import torch
import torch.onnx
from torch.autograd import Variable
from models.retinanet import resnet34, resnet50
from build_network import build_network
input_model = '../user_data/model_data/model_at_epoch_1.dat'
output_dir = 'torch_onnx_model.onnx'
dummy_input = torch.randn(2, 3, 800, 800)
checkpoint = torch.load(input_model)
model, _ = build_network(snapshot=None, backend='retinanet')
model.load_state_dict(checkpoint['state_dict'])
torch.onnx.export(model, dummy_input, output_dir)
print('Done')

the_model = SimpleCNN()
dictionary = torch.load('best_model_dict.pt')
the_model.load_state_dict(dictionary)
the_model.eval()
x = torch.randn(1, 6, 100, 100, requires_grad=True)
torch_out = the_model(x)


torch.onnx.export(the_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "the_model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})