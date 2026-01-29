# tf.random.uniform((B, 1200), dtype=tf.int32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding layer: input_dim=10, output_dim=4, input_length=1200
        # This matches the original example where input shape is (batch, 1200)
        self.embedding = tf.keras.layers.Embedding(
            input_dim=10,
            output_dim=4,
            input_length=1200,
            name='embedding',
            trainable=True
        )
        # Dense layer with linear activation
        self.dense1 = tf.keras.layers.Dense(1, activation='linear')
        # Lambda layer to squeeze last dimension
        self.squeeze = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))
        # Output dense layer, no activation
        self.output_dense = tf.keras.layers.Dense(1, name='output')

    def call(self, inputs, training=False):
        """
        Forward pass:
        inputs: tf.Tensor or dict of inputs with key 'a_input' shape (batch, 1200)
        Output shape: (batch, 1)
        """
        # Inputs could be dict (due to original code) or Tensor, accept both
        if isinstance(inputs, dict):
            x = inputs['a_input']
        else:
            x = inputs

        x = self.embedding(x)          # (B, 1200, 4)
        x = self.dense1(x)             # (B, 1200, 1)
        x = self.squeeze(x)            # (B, 1200)
        x = self.output_dense(x)       # (B, 1)

        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a dict as expected by MyModel: {'a_input': Tensor[int32](B,1200)}
    # Batch size 32 is arbitrary; values between 0 and 9 (input_dim=10)
    batch_size = 32
    input_tensor = tf.random.uniform(
        shape=(batch_size, 1200),
        minval=0,
        maxval=10,
        dtype=tf.int32
    )
    return {'a_input': input_tensor}

