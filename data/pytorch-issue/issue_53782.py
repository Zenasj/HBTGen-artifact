import torchvision

import torch
from torch import onnx
from torchvision.models.resnet import resnet18

def export_resnet_to_onnx(filepath):
    model = resnet18()

    sample_input = torch.randn(16, 3, 224, 224)
    onnx.export(  # resnet18_train_no_params.onnx
        model=model,
        args=sample_input,
        f=filepath.replace('.onnx', '_train_no_params.onnx'),
        export_params=False,
        training=onnx.TrainingMode.TRAINING
    )
    onnx.export(  # resnet18_train_params.onnx
        model=model,
        args=sample_input,
        f=filepath.replace('.onnx', '_train_with_params.onnx'),
        export_params=True,
        training=onnx.TrainingMode.TRAINING
    )

if __name__ == '__main__':
    filepath = 'resnet18.onnx'
    export_resnet_to_onnx(filepath)