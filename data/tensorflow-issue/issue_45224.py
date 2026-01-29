# tf.random.uniform((B, 16, 10), dtype=tf.float32)
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom")
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units, outputs, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.units = units
        self.outputs = outputs

    @property
    def state_size(self):
        return self.units

    def build(self, input_shape):
        self.w = self.add_weight("w", shape=(input_shape[-1], self.units))
        self.o = self.add_weight("o", shape=(self.units, self.outputs))
        self.r = self.add_weight("r", shape=(self.units, self.units))
        self.built = True

    def call(self, inputs, states):
        next_hidden = tf.nn.tanh(
            tf.matmul(inputs, self.w) + tf.matmul(states[0], self.r)
        )
        output = tf.matmul(next_hidden, self.o)
        return output, [next_hidden]

    def get_config(self):
        return {"units": self.units, "outputs": self.outputs}


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define the RNN layer using the custom RNN cell
        self.rnn_layer = tf.keras.layers.RNN(CustomLayer(10, 1), return_sequences=True)

    def call(self, inputs):
        return self.rnn_layer(inputs)


def my_model_function():
    # Return an instance of MyModel with the custom RNN cell inside RNN layer
    return MyModel()


def GetInput():
    # Input shape inferred from example: batch size variable, sequence length=16, features=10
    # Return a random tensor compatible with MyModel input
    # Using batch size 4 as example
    batch_size = 4
    seq_len = 16
    features = 10
    return tf.random.uniform((batch_size, seq_len, features), dtype=tf.float32)

