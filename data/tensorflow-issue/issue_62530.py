# tf.random.uniform((1, 28, 28), dtype=tf.float32) ← Inferred input shape from MNIST images (1 batch, 28x28 grayscale)

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model encapsulates two versions of a MNIST classifier:
    1) External quantization model: expects INT8 quantized inputs and outputs.
    2) Internal quantization model: expects float inputs and embeds quantize/dequantize ops internally.

    The forward pass runs both submodels and returns their output difference,
    illustrating the quantization discrepancy described in the issue.

    Assumptions:
    - Input shape: (batch_size, 28, 28), single channel MNIST images as float32.
    - Both submodels output logits or softmax scores over 10 classes.
    - The models share weights but differ by quantize embedding.
    - For demonstration, the submodels are simplified dense networks mimicking the described models.
    """

    def __init__(self):
        super().__init__()

        # Shared layers (simulate trained weights)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu', name='fc1')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10, name='fc2')
        self.softmax = tf.keras.layers.Softmax(axis=1)

        # Quantize and Dequantize layers for internal quantization simulation
        # Note: In real TFLite internal quant, these are part of the graph.
        # Here we simulate by faking quantize/dequantize in TF ops.

        # Quantize & Dequantize parameters (fake values from the issue)
        self.scale = 1/255.0  # typical scale for quantizing MNIST pixel
        self.zero_point = 0   # fake zero-point

    def call_internal_quant(self, x):
        """
        Simulates internal quantization by embedding quantize & dequantize ops inside the model.
        Input is float32 (0..1), quantized internally to int8, then dequantized.
        """

        # Quantize
        x_int8 = tf.quantization.fake_quant_with_min_max_args(
            x, min=0.0, max=1.0, num_bits=8, narrow_range=False)

        # Pass through network
        x_flat = self.flatten(x_int8)
        x1 = self.dense1(x_flat)
        x1 = self.dropout(x1)
        x2 = self.dense2(x1)
        out = self.softmax(x2)
        return out

    def call_external_quant(self, x):
        """
        Simulates external quantization model: input is quantized INT8 tensor (simulated by int8 cast),
        model assumes quantized input and output, so input must be quantized before feeding.
        We'll quantize input externally before calling this, and output is output logits.
        """

        # Since TensorFlow doesn't process int8 input directly in keras layers,
        # we simulate by dequantizing outside model here for demonstration.
        # In real external quant model, inputs are in int8 and model expects int8 tensors.

        # For demonstration, we mimic external quant by dequantizing inputs here:
        # Assume input x is int8 [-128,127], dequantize to float
        x_float = tf.cast(x, tf.float32) * self.scale  # dequantize

        x_flat = self.flatten(x_float)
        x1 = self.dense1(x_flat)
        x1 = self.dropout(x1)
        x2 = self.dense2(x1)
        # Simulate quantized output by fake quantizing output logits
        # But here we keep float output for comparison
        out = self.softmax(x2)
        return out

    def call(self, inputs):
        """
        Forward pass:
        inputs: float tensor batch (assumed float32, shape (batch,28,28))
        
        Returns:
          - difference = output_internal_quant - output_external_quant
          as float tensor showing discrepancy between two quantization methods.
        """

        # Internal quant model output (simulated)
        out_internal = self.call_internal_quant(inputs)

        # External quant model input requires int8 input, so quantize inputs explicitly
        # Quantize inputs: scale float inputs from [0,1] to int8 range [-128,127]
        inputs_int8 = tf.quantization.fake_quant_with_min_max_args(
            inputs, min=0.0, max=1.0, num_bits=8, narrow_range=False)

        # Cast float fake-quantized inputs to int8 tensor (simulate quantized tensor)
        inputs_int8 = tf.cast((inputs_int8 - 0.5) * 255.0, tf.int8)  # scale roughly

        out_external = self.call_external_quant(inputs_int8)

        # Return difference and bool indicating if outputs match within tolerance
        tolerance = 1e-2
        diff = out_internal - out_external
        match = tf.reduce_all(tf.abs(diff) < tolerance)

        # For clarity, return a dict with outputs and comparison
        return {
            'internal_output': out_internal,
            'external_output': out_external,
            'difference': diff,
            'match': match
        }


def my_model_function():
    """
    Returns an instance of MyModel.
    Weights are randomly initialized here for demonstration,
    but in the original scenario weights are pretrained and identical.
    """
    model = MyModel()

    # Initialize weights by running on dummy input
    _ = model(tf.random.uniform((1, 28, 28), dtype=tf.float32))
    return model


def GetInput():
    """
    Returns a random tensor suitable as input for MyModel.
    The input simulates a batch of 1 MNIST image with pixel values in [0,1].
    """
    return tf.random.uniform((1, 28, 28), minval=0.0, maxval=1.0, dtype=tf.float32)

