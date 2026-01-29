# tf.random.uniform((1, 10, 10, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A simple Conv2D layer as per the original example:
        # input shape: (batch, 10, 10, 3)
        # Conv2D with 1 filter, kernel size 3, bias enabled by default
        self.conv = tf.keras.layers.Conv2D(1, 3, padding='valid', use_bias=True)

        # Note: In TF1.x example, tf.contrib.quantize.create_eval_graph adds fake quant nodes
        # for weights and activations, however min/max nodes for bias are missing.
        # Since tf.contrib.quantize does not exist in TF2, we simulate the quant params here
        # by creating variables representing min/max for weights, bias, and activation.
        # These would normally be learned or calculated during quant-aware training.

        # Create dummy min/max variables to simulate quantization min_max info.
        # Typically these are scalars per tensor:
        self.w_min = tf.Variable(0.0, trainable=False, name='weights_quant_min')
        self.w_max = tf.Variable(6.0, trainable=False, name='weights_quant_max')
        self.b_min = tf.Variable(0.0, trainable=False, name='bias_quant_min')
        self.b_max = tf.Variable(6.0, trainable=False, name='bias_quant_max')
        self.act_min = tf.Variable(0.0, trainable=False, name='act_quant_min')
        self.act_max = tf.Variable(6.0, trainable=False, name='act_quant_max')

    def call(self, inputs, training=False):
        """
        Forward pass:
        - Apply Conv2D
        - For demonstration, output the conv result.
        - In actual quantization flow, fake quantization ops would wrap weights, bias, and output.
        """
        x = self.conv(inputs)

        # Simulate min/max nodes for bias or other tensors (as placeholders),
        # here just returning normal output since TF2 quantization flow is different.
        # The issue is about missing bias quant min/max nodes in TF1 API's create_eval_graph.
        # We'll simply output the conv result.
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return random float32 tensor matching the input shape expected by MyModel
    # Batch size = 1 (minimal usable), height=10, width=10, channels=3 as per original example
    return tf.random.uniform((1, 10, 10, 3), dtype=tf.float32)

