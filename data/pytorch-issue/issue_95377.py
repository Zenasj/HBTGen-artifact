from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image
import requests
import torch

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')

inputs = feature_extractor(images=image, return_tensors="pt")

model = model.to("cuda")
pixel_values = inputs["pixel_values"].to("cuda")

outputs = model(pixel_values=pixel_values)

from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image
import requests
import torch

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')

inputs = feature_extractor(images=image, return_tensors="pt")

model = model.to("cuda")
pixel_values = inputs["pixel_values"].to("cuda")

torch.onnx.export(model, (pixel_values,), "yolosv5.onnx")