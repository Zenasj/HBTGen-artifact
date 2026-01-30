import onnxruntime as ort
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
Channel = 2
class TestInstanceNorm(nn.Module):
    def __init__(self):
        super(TestInstanceNorm, self).__init__()
        self.inorm1 = nn.InstanceNorm2d(Channel, affine=False, track_running_stats=True)
    def forward(self, x):
        y = self.inorm1(x)
        return y

torch.manual_seed(7)
model=TestInstanceNorm()
model.eval()

dummy_input=torch.randn((2,Channel,1,4))
with torch.no_grad():
    torch_out = model(dummy_input)
onnx_model_name="bug_report.onnx"
input_names = [ "actual_input_1" ]
output_names = [ "output1" ]
torch.onnx.export(model, dummy_input, onnx_model_name, export_params=True, verbose=False, input_names=input_names, output_names=output_names, opset_version=11, keep_initializers_as_inputs=True)

ort_session = ort.InferenceSession(onnx_model_name)
onnx_outputs = ort_session.run(None, {'actual_input_1': dummy_input.numpy().astype(np.float32)})
diff_mean = np.mean( np.abs( onnx_outputs[0].reshape(-1)- torch_out.numpy().reshape(-1)))
print("diff_mean=", diff_mean)

print("torch_out=", torch_out)
print("onnx_outputs=", onnx_outputs)

diff_mean= 0.45007646
torch_out= tensor([[[[-0.8201,  0.3956,  0.8989, -1.3884]],

         [[-0.1670,  0.2851, -0.6411, -0.8937]]],


        [[[ 0.9265, -0.5355, -1.1597, -0.4602]],

         [[ 0.7085,  1.0128,  0.2304,  1.0902]]]])
onnx_outputs= [array([[[[-0.6459075 ,  0.68138444,  1.2308291 , -1.2663062 ]],

        [[ 0.41406462,  1.4144142 , -0.63484704, -1.1936316 ]]],


       [[[ 1.6184392 , -0.2994846 , -1.1183207 , -0.20063388]],

        [[-0.15377855,  0.7471721 , -1.5698185 ,  0.97642446]]]],
      dtype=float32)]

track_running_stats=True

nn.InstanceNorm2d

diff_mean= 6.239861e-08
torch_out= tensor([[[[-0.6459,  0.6814,  1.2308, -1.2663]],

         [[ 0.4141,  1.4144, -0.6348, -1.1936]]],


        [[[ 1.6184, -0.2995, -1.1183, -0.2006]],

         [[-0.1538,  0.7472, -1.5698,  0.9764]]]])
onnx_outputs= [array([[[[-0.6459075 ,  0.68138444,  1.2308291 , -1.2663062 ]],

        [[ 0.41406462,  1.4144142 , -0.63484704, -1.1936316 ]]],


       [[[ 1.6184392 , -0.2994846 , -1.1183207 , -0.20063388]],

        [[-0.15377855,  0.7471721 , -1.5698185 ,  0.97642446]]]],
      dtype=float32)]

torch.onnx.export

running_mean

running_var

running_mean

running_var

InstanceNorm2d