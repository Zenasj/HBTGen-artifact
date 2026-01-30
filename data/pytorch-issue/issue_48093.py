import torch
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import onnxruntime as rt
from PIL import Image
import numpy as np
import onnx

shuffle_net = models.shufflenet_v2_x0_5(pretrained=True)
# Export the pytorch model to ONNX
dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True) 
torch.onnx.export(shuffle_net, dummy_input, "shuffleNet.onnx",  opset_version=11, do_constant_folding=True, export_params=True, input_names=['input_images'], output_names=['output']
, dynamic_axes={"input_images": [0], "output": [0]})
# load onnx model
sess = rt.InferenceSession("shuffleNet.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
print('input_name: ' + input_name)
print('label_name: ' + label_name)

img = Image.open('./test.jpg').resize((224, 224))
image_raw_data = np.asarray(img, dtype=np.float32)
image_raw_data = np.transpose(image_raw_data, (2, 0, 1))[np.newaxis, :, :, :]
# random initialize array
images_raw_data = np.ones([1, 3, 224, 224], dtype=np.float32)
# [6, 3, 224, 224]
for i in range(5):
    images_raw_data = np.concatenate((images_raw_data, image_raw_data), axis=0)

pred_onx = sess.run([label_name], {input_name: images_raw_data})
print(pred_onx)