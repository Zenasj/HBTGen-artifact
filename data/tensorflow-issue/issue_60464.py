# tf.random.uniform((B, 32), dtype=tf.float32)  ← Input shape inferred from model input_shape=(32,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the model sequentially like the example:
        # Dense(64, relu) -> Dense(10, linear)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)  # linear activation by default

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel with initialized weights.
    model = MyModel()

    # Build the model by calling with a dummy input
    dummy_input = tf.random.uniform((1, 32), dtype=tf.float32)
    model(dummy_input)  # builds weights

    # Compile the model as per example (optional but included for completeness)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def GetInput():
    # Return a random tensor input matching shape (batch_size, 32)
    # Using batch_size=4 arbitrarily for a reasonable batch input
    return tf.random.uniform((4, 32), dtype=tf.float32)

