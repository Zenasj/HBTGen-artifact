import torch
import torchvision
from PIL import Image
import requests

from transformers import AutoImageProcessor, AutoModelForObjectDetection

with torch.no_grad():
   url = "http://images.cocodataset.org/val2017/000000039769.jpg"
   image = Image.open(requests.get(url, stream=True).raw)

   image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
   model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

   inputs = image_processor(images=image, return_tensors="pt")
   model.eval()

   torch._dynamo.reset()

   compiled = torch.compile(model, dynamic=True, backend="inductor")

   print(compiled(**inputs))