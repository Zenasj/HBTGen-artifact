# tf.random.uniform((batch_size, 88200), dtype=tf.float32)  # Assumed input shape and dtype based on representative dataset

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np

# Custom quantize config from issue for Dense layers with int8 QAT
class DefaultDenseQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    # Using LastValueQuantizer for weights (kernel)
    weights = {
        "kernel": tfmot.quantization.keras.quantizers.LastValueQuantizer(
            num_bits=8, symmetric=True, narrow_range=False, per_axis=False)
    }
    # Using MovingAverageQuantizer for activations
    activations = {
        "activation": tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=8, symmetric=False, narrow_range=False, per_axis=False)
    }

    def get_weights_and_quantizers(self, layer):
        output = []
        for attribute, quantizer in self.weights.items():
            if hasattr(layer, attribute):
                output.append((getattr(layer, attribute), quantizer))
        return output

    def get_activations_and_quantizers(self, layer):
        output = []
        for attribute, quantizer in self.activations.items():
            if hasattr(layer, attribute):
                output.append((getattr(layer, attribute), quantizer))
        return output

    def set_quantize_weights(self, layer, quantize_weights):
        count = 0
        for attribute in self.weights.keys():
            if hasattr(layer, attribute):
                setattr(layer, attribute, quantize_weights[count])
                count += 1

    def set_quantize_activations(self, layer, quantize_activations):
        count = 0
        for attribute in self.activations.keys():
            if hasattr(layer, attribute):
                setattr(layer, attribute, quantize_activations[count])
                count += 1

    def get_output_quantizers(self, layer):
        # No output quantizers by default
        return []

    def get_config(self):
        return {}

# Placeholder CustomLayer referenced in quantize_scope (not defined in the issue)
# We define a pass-through layer for compatibility
class CustomLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs

# The MyModel class is constructed to encapsulate a simple example model combining:
# - A spectrogram extraction layer (simulating YamNet's preprocessing)
# - A dense classification head
# This is a simplification to reflect the context of YamNet and quantization challenge from the issue.

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simulated spectrogram extraction: In YamNet, this would be an STFT or Mel-spectrogram layer.
        # For example purposes, we use a fixed 1D Conv1D layer to represent preprocessing.
        self.spectrogram = tf.keras.layers.Conv1D(
            filters=64, kernel_size=3, strides=1, padding='same', activation='relu'
        )
        # Example dense classification head with quantization config applied
        self.dense = tf.keras.layers.Dense(
            units=521,  # YamNet predicts 521 audio event classes
            activation='softmax'
        )
        # Annotate quantization for the dense layer using DefaultDenseQuantizeConfig
        # Normally done via annotate_layer but here we define quantization in scope.

    def call(self, inputs):
        x = self.spectrogram(inputs)
        # Flatten channel dimension for dense input
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        out = self.dense(x)
        return out

def my_model_function():
    # Build an instance of the model and apply quantization annotation + quantize_apply for QAT
    model = MyModel()

    # Annotate dense layer for quantization
    def apply_quantization_to_layer(layer):
        if isinstance(layer, tf.keras.layers.Dense):
            return tfmot.quantization.keras.quantize_annotate_layer(layer, DefaultDenseQuantizeConfig())
        return layer

    # Clone model to add annotations
    annotated_model = tf.keras.models.clone_model(
        model,
        clone_function=apply_quantization_to_layer,
    )

    # Apply quantization transforms to annotated model for QAT
    qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    # Compile model (same as in issue)
    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return qat_model

def GetInput():
    # From the issue context: YamNet input is audio waveform clips of length 88200 samples at 16 kHz (about 5.5 seconds)
    # Inputs should be float32 tensors shaped (batch_size, 88200)
    # We'll use batch_size=1 for simplicity
    batch_size = 1
    input_length = 88200  # as per representative dataset fixed size in issue
    # Generate random float32 waveform between -1.0 and 1.0 (typical audio range normalized)
    input_tensor = tf.random.uniform(
        shape=(batch_size, input_length),
        minval=-1.0,
        maxval=1.0,
        dtype=tf.float32
    )
    return input_tensor

