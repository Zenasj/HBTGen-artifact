# tf.random.uniform((32, 60, 80), dtype=tf.float32) ← Input shape inferred from example: batch=32, height=60, width=80, single channel assumed

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self,
                 input_shape=(60, 80),
                 n_hidden1=2049,
                 n_hidden2=500,
                 n_hidden3=180,
                 batch_n_mom=0.99,
                 dropout_rate=0.1):
        super().__init__()
        from tensorflow.keras.initializers import he_normal
        # Save input shape for reshape layer
        self.input_shape_ = input_shape
        
        # Build the encoder-decoder (autoencoder-like) model as a sequential stack inside MyModel
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.BatchNormalization(axis=1, momentum=batch_n_mom),

            tf.keras.layers.Dense(n_hidden1, activation="selu", name="he1",
                                  kernel_initializer=he_normal(seed=27)),
            tf.keras.layers.BatchNormalization(axis=1, momentum=batch_n_mom),
            tf.keras.layers.Dropout(dropout_rate),

            tf.keras.layers.Dense(n_hidden2, activation="selu", name="he2",
                                  kernel_initializer=he_normal(seed=42)),
            tf.keras.layers.BatchNormalization(axis=1, momentum=batch_n_mom),

            tf.keras.layers.Dense(n_hidden3, activation="selu", name="he3",
                                  kernel_initializer=he_normal(seed=65)),
            tf.keras.layers.BatchNormalization(axis=1, momentum=batch_n_mom),

            tf.keras.layers.Dense(n_hidden2, activation="selu", name="hd2",
                                  kernel_initializer=he_normal(seed=42)),
            tf.keras.layers.BatchNormalization(axis=1, momentum=batch_n_mom),

            tf.keras.layers.Dense(n_hidden1, activation="selu", name="hd1",
                                  kernel_initializer=he_normal(seed=27)),
            tf.keras.layers.BatchNormalization(axis=1, momentum=batch_n_mom),
            tf.keras.layers.Dropout(dropout_rate),

            tf.keras.layers.Dense(input_shape[0] * input_shape[1], name="output",
                                  kernel_initializer=he_normal(seed=62)),
            tf.keras.layers.Reshape(input_shape)
        ])

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)


def my_model_function():
    """
    Returns a new instance of MyModel with default hyperparameters.
    """
    return MyModel()


def GetInput():
    """
    Returns a random input tensor of shape (32, 60, 80) matching the expected input of MyModel.
    The batch size 32 is used as per the original training example.
    Data type is float32 as typical for neural network inputs.
    """
    return tf.random.uniform(shape=(32, 60, 80), dtype=tf.float32)

