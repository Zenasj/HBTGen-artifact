from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf

class Model(tf.Module):
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(9,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(9)
            ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.MeanSquaredError())
    
    @tf.function(input_signature=[
        tf.TensorSpec([None, 9], tf.float32),
    ])
    def infer(self, x): 
        prediction = self.model(x)
        return prediction

model_custom = Model()
SAVED_MODEL_DIR = "saved_model"

tf.saved_model.save(
    model_custom,
    SAVED_MODEL_DIR,
    signatures={
        "infer":
            model_custom.infer.get_concrete_function(),
        })

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]
converter.experimental_enable_resource_variables = True
converter.allow_custom_ops = True
tflite_model = converter.convert()

py
import torch
import torch.nn as nn
import ai_edge_torch

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )

    def forward(self, x):
        return self.model(x)

# Create the model instance
model = Model()

# Example input that matches the expected input shape of (batch_size, 9)
sample_inputs = (torch.randn(1, 9),)

# Convert the PyTorch model to a TensorFlow Lite model using ai_edge_torch
edge_model = ai_edge_torch.convert(model.eval(), sample_inputs)

# Export the converted model to a .tflite file
edge_model.export("model.tflite")