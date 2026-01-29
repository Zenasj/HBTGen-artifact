# tf.random.uniform((B, None, 100, 100, 3), dtype=tf.float32) ← Assumed input shape for the time-distributed input of model2

import tensorflow as tf

input_shape = (100, 100, 3)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Submodel 1: base Conv2D embedder model
        input1 = tf.keras.layers.Input(shape=input_shape)
        cov = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, name='cov1')(input1)
        self.embedding_model = tf.keras.Model(inputs=input1, outputs=cov, name='model1')

        # Submodel 2: sequence model using TimeDistributed wrapping model1
        input_seq = tf.keras.layers.Input(shape=(None,) + input_shape)  # (batch, time, H, W, C)
        time_distributed = tf.keras.layers.TimeDistributed(self.embedding_model, name='time_dis1')
        emb = time_distributed(input_seq)
        att = tf.keras.layers.Attention()([emb, emb])
        dense1 = tf.keras.layers.Dense(64, name='dense1')(att)
        outputs = tf.keras.layers.Softmax(axis=-1)(dense1)
        self.model2 = tf.keras.Model(inputs=input_seq, outputs=outputs, name='model2')

    def call(self, inputs):
        """
        Forward through model2 (the sequence model).
        Inputs: tensor of shape (batch_size, time_steps, 100, 100, 3)
        Returns: softmax output of shape (batch_size, time_steps, 98, 98, 64)
        """
        return self.model2(inputs)

    def call_with_intermediate(self, inputs):
        """
        Returns outputs of intermediate layers:
          - output of model1.conv layer for the frames within model2 (time_dis1.layer.cov1 output)
          - output of dense1 layer in model2

        inputs: tensor of shape (batch_size, time_steps, 100, 100, 3)

        returns: tuple (conv_out, dense1_out)
        conv_out shape: (batch_size, time_steps, h1, w1, filters) -> from Conv2D (98,98,32)
        dense1_out shape: (batch_size, time_steps, 98, 98, 64)
        """
        # Apply embedding_model time-distributed to sequence
        emb = self.model2.get_layer('time_dis1')(inputs)
        # Attention on embeddings
        att = self.model2.get_layer('attention')([emb, emb])
        dense_output = self.model2.get_layer('dense1')(att)

        # The conv output from model1 for each time step (already computed in emb)
        # emb has shape (batch, time, h, w, filters)
        conv_output = emb

        return conv_output, dense_output

def my_model_function():
    """
    Returns an instance of MyModel with the composed sub-models.
    """
    return MyModel()

def GetInput():
    """
    Generates a random tensor mimicking the input expected by MyModel/model2:
    Shape: (batch, time_steps, height, width, channels)
    dtype: float32
    Assumed batch=1, time_steps=10 for demonstration.
    """
    import tensorflow as tf
    return tf.random.uniform((1, 10, 100, 100, 3), dtype=tf.float32)

