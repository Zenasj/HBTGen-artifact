# torch.rand(1, 3, 224, 224, dtype=torch.float32)  # Input shape inferred from prepare step
import torch
import torch.nn as nn
from torchvision import models
from torch.ao.quantization import (
    QConfigMapping,
    quantize_fx,
    MinMaxObserver,
    default_per_channel_weight_observer,
)
from torchvision.models.resnet import ResNet50_Weights

class MyModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # Quantized ResNet50 model
        
    def forward(self, x):
        return self.model(x)

def my_model_function():
    # Initialize and quantize ResNet50 with S8S8 per-channel configuration
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT).eval()
    
    # Define QConfig for S8S8 per-channel quantization
    qconfig = torch.ao.quantization.QConfig(
        activation=MinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=True
        ),
        weight=default_per_channel_weight_observer
    )
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    
    # Prepare for quantization
    example_input = torch.randn(1, 3, 224, 224)
    model_prepared = quantize_fx.prepare_fx(model, qconfig_mapping, example_input)
    
    # Dummy calibration step (replaces missing calib_data_loader)
    dummy_data = [torch.randn(1, 3, 224, 224) for _ in range(5)]
    for data in dummy_data:
        model_prepared(data)
    
    # Convert to quantized model
    model_quantized = quantize_fx.convert_fx(model_prepared)
    
    return MyModel(model_quantized)  # Wrap quantized model in MyModel

def GetInput():
    # Generate valid input matching ResNet50's expected input shape
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

