import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
import io
import onnx

model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
device = "cuda:0"
pt_model = model.to(device)

batch_size, channel, input_size_h, input_size_w = 64, 3, 300, 300
images = torch.randn(batch_size, channel, input_size_h, input_size_w, device=device)

num_boxes = 5
targets = []
for _ in range(batch_size):
    boxes = torch.rand(num_boxes, 4, device=device) * 300
    boxes[:, 2:] += boxes[:, :2]
    target = {
        'boxes': boxes,
        'labels': torch.randint(1, 91, (num_boxes,), device=device)
    }
    targets.append(target)

model_inputs = (images, targets)
input_names = ["input", "targets"]
output_names = ["output"]
dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

f = io.BytesIO()
torch.onnx.export(
    pt_model,
    model_inputs,
    f,
    input_names=input_names,
    output_names=output_names,
    opset_version=18,
    do_constant_folding=False,
    training=torch.onnx.TrainingMode.TRAINING,
    dynamic_axes=dynamic_axes,
    export_params=True,
    keep_initializers_as_inputs=False,
)

onnx_model = onnx.load_model_from_string(f.getvalue())