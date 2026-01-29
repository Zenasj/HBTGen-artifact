# tf.random.uniform((B, 10), dtype=tf.float32)  ← Input shape is (batch_size, 10)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple model similar to example:
        # Input (10) -> Dense(10, kernel_initializer='ones') -> Dense(1)
        self.dense1 = tf.keras.layers.Dense(10, kernel_initializer='ones')
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        outputs = self.dense2(x)
        # Add losses similarly to original example:
        # 1) mean of outputs as a loss
        output_loss = tf.reduce_mean(outputs)
        self.add_loss(output_loss)

        # 2) sum of the sizes of trainable weights as a scalar loss
        reg_losses = [tf.cast(tf.size(w), tf.float32) for w in self.trainable_weights]
        reg_loss = tf.add_n(reg_losses)
        self.add_loss(reg_loss)

        return outputs

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile with loss=None since losses are added manually
    model.compile(optimizer="adam", loss=[None] * 1)
    return model

def GetInput():
    # Return a random input tensor matching the model's expected input shape
    # Here batch size B = 2 (based on example), input shape = (10,)
    return tf.random.uniform((2, 10), dtype=tf.float32)

