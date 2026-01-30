import torch
import numpy as np

dummy_input = torch.randn(1, 3, 736, 1312, device='cpu', requires_grad=True)
inputs = ['input']
outputs = ['output']
dynamic_axes = {'input':{1:'height', 2:'width'},
                           'output':{1:'height', 2:'width'}}
torch.onnx.export(predictor.model.eval(), dummy_input, "FPNInception.onnx",
                             export_params=True, do_constant_folding=True,
                             input_names=inputs, output_names=outputs, dynamic_axes=dynamic_axes,
                             opset_version=12, verbose=False)

#summary_input is a copy input tensor with preprocessing from Predictor class
import onnxruntime
ort_session = onnxruntime.InferenceSession('data/'+n_model+'/'+n_model+'.onnx')
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
inputs = [node for node in ort_session.get_inputs()][0]
outputs = [node.name for node in ort_session.get_outputs()]
y_pred = predictor.model.eval()(*summary_input)
pred_onx = ort_session.run(outputs, {inputs.name: to_numpy(*summary_input)})
output = pred_onx [0]

out = np.transpose(output[0,:,:,:], (1, 2, 0))
print(np.max(out), '   ', np.min(out))
out += 1
out = out / 2  * 255
out = out[:h,:w,:].astype('uint8')
out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
plt.imshow(out)
plt.title('output')
plt.show()