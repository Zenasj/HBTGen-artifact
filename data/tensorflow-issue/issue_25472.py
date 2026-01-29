# tf.random.uniform((B, 1), dtype=tf.float32) ← Inferring input shape as (batch_size, 1) based on the example with Dense(1, input_shape=(1,))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # replicate the single Dense layer model from example
        self.dense = tf.keras.layers.Dense(1, input_shape=(1,))
        # Adam optimizer - mimicking keras optimizer in example
        self.optimizer = tf.keras.optimizers.Adam()

    def call(self, inputs, training=False):
        # Simple forward pass
        return self.dense(inputs)

    @tf.function
    def train_step(self, inputs):
        # Custom train step using GradientTape to replace deprecated get_updates()
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = tf.keras.losses.mse(inputs, predictions)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

def my_model_function():
    # Return an instance of the model
    return MyModel()

def GetInput():
    # Input compatible with model: random float tensor of shape (B, 1)
    # Batch size is arbitrarily chosen as 4 for testing
    return tf.random.uniform((4, 1), dtype=tf.float32)

