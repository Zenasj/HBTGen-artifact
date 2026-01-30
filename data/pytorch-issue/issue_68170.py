import torch

dummy_input_1 = torch.randn(1, seq_length, requires_grad=True).long()
dummy_input_2 = torch.randn(seq_length, requires_grad=True).long()

dynamic_axes = {'input_1' : {1 : 'len'}, 'input_2' : {0 : 'len'}, 
                'output1': {0: 'label'}} 

torch.onnx.export(model,
        args=(dummy_input_1, dummy_input_2),
        f='model.onnx',
        input_names=['input_1', 'input_2'],
        output_names=['output1'],
        dynamic_axes=dynamic_axes)