import torch

class PaddingLayer(torch.jit.ScriptModule):
    def __init__(self):
        super(PaddingLayer, self).__init__()
    
    @torch.jit.script_method
    def forward(self,input_t, shape_t):
        # type: (Tensor, Tensor) -> Tensor
        output_size = [len(shape_t),int(shape_t.max())]
        model_input = torch.zeros(size=output_size,dtype=input_t.dtype, device = input_t.device)
        s_start=0
        for s_id,s_len in enumerate(shape_t):
            model_input[s_id,:s_len] = input_t[s_start:s_start+s_len]
            s_start = s_start+s_len
        return model_input
    
    
input_t = torch.ones(size=[10],dtype=torch.long)
shape_t = torch.tensor([1,2,3,4],dtype=torch.long)

p = torch.jit.script(PaddingLayer())
example_output = p(input_t,shape_t)

print(example_output)


torch.onnx.export(p,
                 (input_t, shape_t),
                  "test_tracer.onnx", # where to save the model (can be a file or file-like object)
                  export_params=True,
                  opset_version=11,
                  input_names=['input','shape_tensor'],
                  output_names=['padded_input'],
                  dynamic_axes={'input': {0: 'concat_sequence'},
                                'shape_tensor': {0: 'number_of_seq'},
                                'padded_input': {0: 'number_of_seq', 1:'max_padded_sequence'}},
                  example_outputs = [example_output]
                  
                  
                 )

tensor([[1, 0, 0, 0],
          [1, 1, 0, 0],
          [1, 1, 1, 0],
          [1, 1, 1, 1]])

import onnxruntime
sess = onnxruntime.InferenceSession('test_tracer.onnx')
ort_out = sess.run(None, {'input': input_t.numpy(), 'shape_tensor': shape_t.numpy()})
print(example_output)
print(ort_out)