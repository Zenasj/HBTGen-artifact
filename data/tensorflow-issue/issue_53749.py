# tf.random.uniform((1, 28, 28, 1), dtype=tf.float32) ← Inferred input shape and type from the keras.Input(shape=(28, 28, 1)) 

import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot

LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer

class DefaultConv2DQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    """Custom QuantizeConfig to set weights quantized to 8 bits and activations to 16 bits for Conv2D layers."""
    def get_weights_and_quantizers(self, layer):
        # Quantize conv kernel weights with 8 bits, symmetric, per-tensor quantization.
        return [(layer.kernel, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]
    
    def get_activations_and_quantizers(self, layer):
        # Quantize activations with 16 bits, asymmetric, moving average quantizer.
        return [(layer.activation, MovingAverageQuantizer(num_bits=16, symmetric=False, narrow_range=False, per_axis=False))]
    
    def set_quantize_weights(self, layer, quantize_weights):
        layer.kernel = quantize_weights[0]
    
    def set_quantize_activations(self, layer, quantize_activations):
        layer.activation = quantize_activations[0]
    
    def get_output_quantizers(self, layer):
        # No additional output quantization specified
        return []
    
    def get_config(self):
        # Required for serialization; keep empty or add parameters if needed
        return {}

def apply_quantization_to_conv2d(layer):
    # Annotate Conv2D layers with the custom QuantizeConfig to have mixed-bit quantization as specified.
    if isinstance(layer, tf.keras.layers.Conv2D):
        return tfmot.quantization.keras.quantize_annotate_layer(layer, DefaultConv2DQuantizeConfig())
    return layer

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define input layer explicitly
        self.input_layer = keras.layers.InputLayer(input_shape=(28, 28, 1))
        
        # Define Conv2D layers with ReLU activation; these will be quantized by the QuantizeConfig
        self.conv1 = keras.layers.Conv2D(filters=12, kernel_size=(3,3), activation='relu')
        self.conv2 = keras.layers.Conv2D(filters=12, kernel_size=(3,3), activation='relu')
        self.conv3 = keras.layers.Conv2D(filters=12, kernel_size=(3,3), activation='relu')

        # After defining, annotate layers for quantization
        # The quantize_annotate_layer is typically called outside model class during model cloning,
        # so we will do it inside call here for simplicity in this example.

    def call(self, inputs, training=False):
        # Forward pass through conv layers
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

def my_model_function():
    # Build the Keras model with quantize annotation and apply quantization transforms
    
    # Create a functional model first
    inputs = keras.Input(shape=(28,28,1))
    x1 = keras.layers.Conv2D(filters=12, kernel_size=(3,3), activation='relu')(inputs)
    x2 = keras.layers.Conv2D(filters=12, kernel_size=(3,3), activation='relu')(x1)
    x3 = keras.layers.Conv2D(filters=12, kernel_size=(3,3), activation='relu')(x2)
    base_model = keras.Model(inputs, x3)
    
    # Annotate conv2d layers with the custom QuantizeConfig
    def annotate(layer):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return tfmot.quantization.keras.quantize_annotate_layer(layer, DefaultConv2DQuantizeConfig())
        return layer

    annotated_model = tf.keras.models.clone_model(base_model, clone_function=annotate)
    
    # Apply quantization transforms in the quantize_scope so the custom QuantizeConfig is recognized
    with tfmot.quantization.keras.quantize_scope({'DefaultConv2DQuantizeConfig': DefaultConv2DQuantizeConfig}):
        quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    
    # Wrap the quant-aware model as an instance of tf.keras.Model for API consistency
    # (Optionally could subclass and incorporate quantization internally, but tfmot typically works this way)
    
    return quant_aware_model

def GetInput():
    # Return a random tensor with batch size 1 matching input shape (28, 28, 1)
    # dtype float32 as per the original model input
    return tf.random.uniform(shape=(1, 28, 28, 1), dtype=tf.float32)

