# tf.random.uniform((1, 2, 4, 8), dtype=tf.float32)
import tensorflow as tf
import tensorflow_model_optimization as tfmot

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Two ReLU layers applied independently to two inputs
        self.relu1 = tf.keras.layers.ReLU()
        self.relu2 = tf.keras.layers.ReLU()
        # Concatenate along channels (last) dimension
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, inputs):
        inp1, inp2 = inputs
        r1 = self.relu1(inp1)
        r2 = self.relu2(inp2)
        c1 = self.concat([r1, r2])
        return c1

def my_model_function():
    """
    Constructs the MyModel instance and applies QAT quantization using 16-bit activations and 8-bit weights.
    This matches the original model structure and quantization scheme in the reported issue.
    """
    model = MyModel()

    # To replicate the quantization aware training setup with 16-bit activation and 8-bit weight quantization,
    # we'll build a functional Keras model wrapping MyModel, annotate it, and apply the quantization scheme.
    input1 = tf.keras.Input(shape=(2, 4, 8), batch_size=1, name='input1')
    input2 = tf.keras.Input(shape=(2, 4, 8), batch_size=1, name='input2')
    outputs = model([input1, input2])
    functional_model = tf.keras.Model(inputs=[input1, input2], outputs=outputs)

    scheme_16_8 = tfmot.quantization.keras.experimental.default_n_bit.DefaultNBitQuantizeScheme(
        disable_per_axis=False, num_bits_weight=8, num_bits_activation=16)

    # Annotate and apply quantization aware training scheme
    annotated_model = tfmot.quantization.keras.quantize_annotate_model(functional_model)
    q_model = tfmot.quantization.keras.quantize_apply(annotated_model, scheme=scheme_16_8)

    return q_model

def GetInput():
    """
    Returns a tuple of two random tensors with dtype float32 and shape that matches the model input:
    batch_size=1, height=2, width=4, channels=8.
    """
    inp1 = tf.random.uniform((1, 2, 4, 8), dtype=tf.float32)
    inp2 = tf.random.uniform((1, 2, 4, 8), dtype=tf.float32)
    return (inp1, inp2)

