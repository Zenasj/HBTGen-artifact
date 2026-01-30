import torch.nn as nn

import torch
import onnx

class WrapFunction(torch.nn.Module):

    def __init__(self, wrapped_function):
        super(WrapFunction, self).__init__()
        self.wrapped_function = wrapped_function

    def forward(self, *args, **kwargs):
        return self.wrapped_function(*args, **kwargs)
    
def test_flatten(boxes, scores, selected_indices):
    batch_inds, cls_inds = selected_indices[:, 0], selected_indices[:, 1]
    box_inds = selected_indices[:, 2]
    boxes = boxes[batch_inds, box_inds, :]
    scores = scores[batch_inds, cls_inds, box_inds]
    dets = torch.cat([boxes, scores[:, None]], dim=1)
    return dets, batch_inds, cls_inds

# create input data
batch_size = 2
num_box = 10
num_class = 2
num_det = 5
boxes = torch.rand(batch_size, num_box, 4) * 10
scores = torch.rand(batch_size, num_class, num_box) * 10
batch_inds = torch.randint(batch_size, (num_box, ))
cls_inds = torch.randint(num_class, (num_box, ))
box_inds = torch.randint(num_box, (num_box, ))
selected_indices = torch.cat([batch_inds[:, None], cls_inds[:, None], box_inds[:, None]], dim=1)
input_data = (boxes, scores, selected_indices)

wrapped_model = WrapFunction(test_flatten)
wrapped_model.cpu().eval()
onnx_file = 'test_flatten.onnx'
with torch.no_grad():
    torch.onnx.export(
        wrapped_model, 
        input_data,
        onnx_file,
        export_params=True,
        keep_initializers_as_inputs=True,
        input_names=['boxes', 'scores', 'selected_indices'],
        output_names=['dets', 'batch_indices', 'labels'],
        opset_version=11)
onnx_model = onnx.load(onnx_file)

# run with onnxruntime
sess = ort.InferenceSession(onnx_file)
onnx_outputs = sess.run(None, {
    'scores': scores.detach().numpy(),
    'boxes': boxes.detach().numpy(),
    'selected_indices':selected_indices.detach().numpy()
})