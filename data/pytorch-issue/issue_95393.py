from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import onnxruntime as ort
import numpy as np
import torch.onnx
import os
import timeit
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
#import pdb; pdb.set_trace()
def torch_inference(inputs):
    outputs = model(**inputs)
    return outputs
outputs = torch_inference(inputs)
logits_per_image = outputs.logits_per_image
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits_per_image.argmax(-1).item()
options = ort.SessionOptions()
# options.log_severity_level = 0
if not os.path.exists("clip.onnx"):
    torch.onnx.export(model, inputs, "clip.onnx", export_params=True, opset_version=11, do_constant_folding=True, input_names=["input"], output_names=["output"])
ort_sess = ort.InferenceSession("clip.onnx", providers=["CUDAExecutionProvider"], sess_options=options)
ort_inputs = {"input":inputs['pixel_values'].numpy()}
def ort_inference(inputs):
  ort_outputs=ort_sess.run(None, inputs)
  return ort_outputs
ort_outputs = ort_inference(ort_inputs)
ort_prediction=int(np.argmax(np.array(ort_outputs[0]).squeeze(), axis=0))
if ort_prediction == predicted_class_idx:
    print("Test passed")
else:
    print("Test failed")
print(model.config.id2label[predicted_class_idx])
print(model.config.id2label[ort_prediction])
print(timeit.repeat(lambda: torch_inference(inputs), repeat=1, number=1000))
print(timeit.repeat(lambda: ort_inference(ort_inputs), repeat=1, number=1000))

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import onnxruntime as ort
import numpy as np
import torch.onnx
import os
import timeit
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
#import pdb; pdb.set_trace()
def torch_inference(inputs):
    outputs = model(**inputs)
    return outputs
outputs = torch_inference(inputs)
logits_per_image = outputs.logits_per_image
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits_per_image.argmax(-1).item()
options = ort.SessionOptions()
# options.log_severity_level = 0
if not os.path.exists("clip.onnx"):
    torch.onnx.export(model, (inputs.input_ids, inputs.pixel_values, inputs.attention_mask), "clip.onnx", export_params=True, opset_version=17, do_constant_folding=True, input_names=["input_ids", "pixel_values", "attention_mask"], output_names=["output"])
ort_sess = ort.InferenceSession("clip.onnx", providers=["CUDAExecutionProvider"], sess_options=options)
#ort_inputs = {"input":inputs['pixel_values'].numpy()}
ort_inputs = {"input_ids": inputs.input_ids.numpy(), "pixel_values": inputs.pixel_values.numpy(), "attention_mask": inputs.attention_mask.numpy()}
def ort_inference(inputs):
  ort_outputs=ort_sess.run(None, inputs)
  return ort_outputs
ort_outputs = ort_inference(ort_inputs)
ort_prediction=int(np.argmax(np.array(ort_outputs[0]).squeeze(), axis=0))
if ort_prediction == predicted_class_idx:
    print("Test passed")
else:
    print("Test failed")
print(model.config.id2label[predicted_class_idx])
print(model.config.id2label[ort_prediction])
#print(timeit.repeat(lambda: torch_inference(inputs), repeat=1, number=1000))
#print(timeit.repeat(lambda: ort_inference(ort_inputs), repeat=1, number=1000))