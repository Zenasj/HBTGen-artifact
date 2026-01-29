# tf.random.uniform((B, 2), dtype=tf.float32) ← Input shape inferred from train_features with 2 columns: 'a' and 'b'

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Normalization layer as in the original model
        self.normalizer = tf.keras.layers.Normalization()
        # Weights of the normalizer will be set by adapt() later
        
        # Dense layers following the example model
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1)

    def adapt(self, data):
        # Adapt the normalization layer with training data (numpy array or tensor)
        self.normalizer.adapt(data)

    def call(self, inputs, training=False):
        x = self.normalizer(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)


def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Adapt the normalizer with a small sample of data for demo purposes
    # In practice, adapt should be called with training data prior to training/saving
    example_data = tf.constant([[20.0, 5.0], [20.0, 10.0]], dtype=tf.float32)
    model.adapt(example_data)
    return model


def GetInput():
    # Return a random tensor input that matches expected input shape for MyModel
    # Since the input features are 'a' and 'b' only, shape=(batch_size, 2)
    # We'll generate a batch of 4 samples with float32 values roughly in range [0, 30] (based on example data)
    batch_size = 4
    input_shape = (batch_size, 2)
    # Uniformly sample between 0 and 30
    return tf.random.uniform(input_shape, minval=0, maxval=30, dtype=tf.float32)

