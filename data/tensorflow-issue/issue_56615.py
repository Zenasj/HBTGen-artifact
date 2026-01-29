# tf.random.uniform((1, 64, 64, 3), dtype=tf.float32)
import tensorflow as tf

def quant(x, num_bits=8):
    """
    Fake quantization function mimicking quantization to num_bits bits.
    Used for weights and activations.
    """
    return tf.quantization.fake_quant_with_min_max_args(x, -1, 1, num_bits, narrow_range=False)

class QConv(tf.keras.layers.Conv2D):
    """
    Quantized convolution layer that applies fake quantization to weights and activation outputs.
    The activation quantizer can produce int8 or int16 like outputs via fake quanitzation.
    """
    def __init__(self, filters, kernel_size, weight_quantizer, activation_quantizer):
        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer
        super().__init__(filters=filters, kernel_size=kernel_size)

    def call(self, inputs):
        # Quantize the kernel weights
        quantized_kernel = self.weight_quantizer(self.kernel)
        # Perform convolution with quantized weights
        conv_output = self.convolution_op(inputs, quantized_kernel)
        # Quantize the activation output
        return self.activation_quantizer(conv_output)

class MyModel(tf.keras.Model):
    """
    Model that contains two branches sharing the same quantized input:
    - one convolution producing int8-like output
    - another convolution producing int16-like output (via fake quantization)
    Outputs both tensors.
    """
    def __init__(self):
        super().__init__()
        # Quantizer for weights and input activations is 8-bit for both convolutions
        self.weight_quantizer = lambda x: quant(x, num_bits=8)
        # Activation quantizer producing int8-like fake quantization output
        self.activation_quantizer_8 = lambda x: quant(x, num_bits=8)
        # Activation quantizer producing int16-like fake quantization output
        self.activation_quantizer_16 = lambda x: quant(x, num_bits=16)

        # Two separate convolution layers for 8bit and 16bit output branches
        self.conv8 = QConv(filters=32, kernel_size=3,
                           weight_quantizer=self.weight_quantizer,
                           activation_quantizer=self.activation_quantizer_8)
        self.conv16 = QConv(filters=32, kernel_size=3,
                            weight_quantizer=self.weight_quantizer,
                            activation_quantizer=self.activation_quantizer_16)

    def call(self, inputs):
        # Inputs expected to be float32 tensors (fake quant input)
        x = quant(inputs, num_bits=8)  # Quantize input to 8-bit before convolution (mimicking input quant)
        out8 = self.conv8(x)
        out16 = self.conv16(x)
        # Return both outputs similar to the example
        return out8, out16

def my_model_function():
    # Returns an instance of MyModel with initialized weights
    return MyModel()

def GetInput():
    # Return a random float32 tensor of shape (1, 64, 64, 3) representing an input image batch
    # The input shape and dtype matches the example's input_tensor
    return tf.random.uniform((1, 64, 64, 3), minval=-1, maxval=1, dtype=tf.float32)

