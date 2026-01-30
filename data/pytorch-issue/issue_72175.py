torch.onnx.export(
        model,
        input,
        "test.onnx",
        input_names=["input"],
        output_names=["output"],
        export_params=True,
    )

import torchvision.models as models
model = models.resnet18()

import torch
import torch.nn as nn

class SVHN(nn.Module):
    def __init__(self):
        super().__init__()
        conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        bn = nn.BatchNorm2d(32, affine=False)
        relu = nn.ReLU()
        dropout = nn.Dropout(0.3)
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.features = nn.Sequential(
            conv,
            bn,
            relu,
            dropout,
            maxpool,
        )
        self.classifier = nn.Sequential(nn.Linear(8192, 10))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        x = self.softmax(x)
        return x

# load model
model = SVHN()
model = model.eval()
model = model.cuda()
model = model.half()

# create sample input
input = torch.Tensor(1, 3, 32, 32)
input = input.cuda()
input = input.half()

# attempt export to ONNX
torch.onnx.export(
    model,
    input,
    "test.onnx",
    input_names=["input"],
    output_names=["output"],
    export_params=True,
)

# change onnx input/output types
import onnx
onnx_model = onnx.load(ONNX_FILE_PATH)
graph = onnx_model.graph
in_type = getattr(graph.input[0], "type", None)
getattr(in_type, "tensor_type", None).elem_type = 10  # fp16
out_type = getattr(graph.output[0], "type", None)
getattr(out_type, "tensor_type", None).elem_type = 10  # fp16
onnx.save(onnx_model, ONNX_FILE_PATH)