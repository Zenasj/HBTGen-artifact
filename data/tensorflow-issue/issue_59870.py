import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(in_keras_path)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]
converter.target_spec.experimental_supported_backends = ["GPU"] # if empty, GPU is not enabled

converter.experimental_new_converter = True

converter.optimizations = [tf.lite.Optimize.DEFAULT] #8-bit quantization
converter.allow_custom_ops = True

tflite_quant_model = converter.convert()

py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
  
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.batch_to_space = nn.BatchNorm2d(16)  # Example batch normalization layer
        self.space_to_batch = nn.ReLU()  # Example activation layer
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        

        self.flatten = nn.Flatten()
        
       
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # Adjust input size based on your data shape
        self.fc2 = nn.Linear(128, 10)  # Output layer with 10 classes

    def forward(self, x):
        
        
  
        x = self.conv1(x)
        x = F.relu(x)
        
  
        x = self.batch_to_space(x)
        
       
        x = self.space_to_batch(x)
        
     
        x = self.conv2(x)
        x = F.relu(x)
        
     
        x = self.flatten(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x


model = MyModel()


input_tensor = torch.randn(1, 3, 32, 32)

edge_model = ai_edge_torch.convert(efficientnet_model.eval(), (input_tensor,))
edge_model.export("simple_conv_.tflite")