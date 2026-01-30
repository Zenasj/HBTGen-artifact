from tensorflow.keras import layers
from tensorflow.keras import models

import tf_keras as keras
model.save(dP.model_name)

def convertModelToTFLite(model_file):
    import tensorflow as tf
    import tf_keras as keras
    model = keras.models.load_model(model_file)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)    # TensorFlow 2.15 and earlier
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)
    open(convFile, "wb").write(tflite_model)

model = keras.models.load_model(model_name)

import keras
model.export(dP.model_name)

model = keras.models.Sequential()
model.add(keras.layers.TFSMLayer(model_name, call_endpoint='serve'))

def convertModelToTFLite(model_file):
    import tensorflow as tf
    import keras
    model = keras.layers.TFSMLayer(model_file, call_endpoint='serve') # TensorFlow >= 2.16.0
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
   
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)
    open(convFile, "wb").write(tflite_model)

py
from types import SimpleNamespace

import ai_edge_torch
import torch
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self, input_shape, num_classes, params):
        super(CustomModel, self).__init__()
        self.params = params
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        for i in range(len(params.channels)):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(input_shape[1] if i == 0 else params.channels[i-1], params.channels[i], (1, params.kernel_widths[i])),
                nn.ReLU(),
                nn.MaxPool2d((1, params.pooling_widths[i])) if i < len(params.pooling_widths) else nn.Identity(),
                nn.Dropout(params.conv_dropout_rate)
            ))

        # Calculate the size of flattened features
        with torch.no_grad():
            x = torch.randn(*input_shape)
            for layer in self.conv_layers:
                x = layer(x)
            flattened_size = x.numel()

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        for i in range(len(params.hidden_layer_neurons)):
            self.fc_layers.append(nn.Sequential(
                nn.Linear(flattened_size if i == 0 else params.hidden_layer_neurons[i-1], params.hidden_layer_neurons[i]),
                nn.ReLU(),
                nn.Dropout(params.fc_dropout_rate)
            ))

        # Output layer
        if params.regressor:
            self.output_layer = nn.Linear(params.hidden_layer_neurons[-1], 1)
        else:
            self.output_layer = nn.Linear(params.hidden_layer_neurons[-1], num_classes)
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.fc_layers:
            x = layer(x)
        x = self.output_layer(x)
        if not self.params.regressor:
            x = self.softmax(x)
        return x


params = SimpleNamespace()
params.channels = [6, 12, 24]
params.kernel_widths = [3, 4, 2]
params.pooling_widths = [3, 4, 2]
params.conv_dropout_rate = 0.2
params.hidden_layer_neurons = [128, 256, 128]
params.fc_dropout_rate = 0.2
params.regressor = False


model = CustomModel((1, 3, 224, 224), 10, params)
sample_input = (torch.randn(1, 3, 224, 224),)

edge_model = ai_edge_torch.convert(model.eval(), sample_input)
edge_model.export("custom_model.tflite")