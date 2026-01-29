# tf.random.uniform((1, 4, 3), dtype=tf.float32) ← inferred input shape from Input(shape=([4,3]), batch_size=1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers corresponding to the example model with two outputs
        self.dense1 = tf.keras.layers.Dense(20)
        self.output1_layer = tf.keras.layers.Dense(1, name="output1")
        self.output2_layer = tf.keras.layers.Dense(1, name="output2")

    def call(self, inputs):
        x = self.dense1(inputs)
        output1 = self.output1_layer(x)
        output2 = self.output2_layer(x)
        return [output1, output2]

def my_model_function():
    model = MyModel()
    # Compile with binary crossentropy losses and loss_weights, carefully ensuring loss_weights are numeric (not tf.Variable)
    # To avoid the serialization error, loss_weights must be numeric types, not tf.Variable or tf.keras.backend.variable.
    alpha_value = 0.25  # Replace tf.keras.backend.variable(0.25) with plain float for serialization compatibility
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "output1": tf.keras.losses.binary_crossentropy,
            "output2": tf.keras.losses.binary_crossentropy
        },
        loss_weights=[1.0, alpha_value]
    )
    return model

def GetInput():
    # Return a sample input tensor matching input shape (batch_size=1, 4, 3) with float32 dtype
    return tf.random.uniform((1, 4, 3), dtype=tf.float32)

