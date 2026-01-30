import numpy as np
import onnxruntime as ort
import torch
import torchvision
import urllib

from torch.onnx import TrainingMode
from torchvision import transforms
from PIL import Image

ssdlite = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
ssdlite = ssdlite.eval()

try:
    urllib.URLopener().retrieve("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
except:
    urllib.request.urlretrieve("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")

input_image = Image.open("dog.jpg")
preprocess = transforms.Compose([
    transforms.Resize(352),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

with torch.no_grad():
    torch.onnx.export(ssdlite, input_batch, "ssdlite320_mobilenet_v3_large.onnx",
                      input_names=['image'], output_names=['boxes', 'scores', 'labels'],
                      opset_version=12,
                      do_constant_folding=False,
                      training=TrainingMode.EVAL,
                      export_params=True,
                      keep_initializers_as_inputs=False
                      )

print("Running model with pytorch")
torch_results = ssdlite(input_batch)
print(torch_results)

s = ort.InferenceSession("ssdlite320_mobilenet_v3_large.onnx")
print("Running ONNX model with dog.jpg")
np_input = input_batch.detach().cpu().numpy()
ort_results = s.run(None, {'image': np_input})
print(ort_results)

print("Running ONNX model with zeros")
np_zeros_input = np.zeros((1, 3, 320, 320), np.float32)
zeros_results = s.run(None, {'image': np_zeros_input})
print(zeros_results)