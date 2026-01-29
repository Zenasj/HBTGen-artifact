# tf.random.uniform((B, 4), dtype=tf.float32)  # Input shape inferred from iris dataset features (4 features)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the model architecture from the provided Sequential example
        self.dense1 = tf.keras.layers.Dense(16, activation='relu', input_shape=(4,))
        self.dense2 = tf.keras.layers.Dense(8, activation='relu')
        self.dense3 = tf.keras.layers.Dense(8, activation='relu')
        self.out = tf.keras.layers.Dense(3, activation='softmax')  # 3 classes for iris

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.out(x)

def my_model_function():
    # Return an instance of MyModel, compiled similar to the original model
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )
    return model

def GetInput():
    # Generate dummy input matching iris features: batch size 32, 4 features, float32
    # Use uniform random values as normalized input approximation
    return tf.random.uniform((32, 4), dtype=tf.float32)

