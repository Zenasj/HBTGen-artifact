from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import requests
import torch.onnx

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "How many cats are there?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

inputs = processor(image, text, return_tensors="pt")

import torch._dynamo

gm, _ = torch._dynamo.export(model, **inputs, aten_graph=True, tracing_mode="real")
gm.print_readable()