# tf.random.uniform((64, 1), dtype=tf.float32) ← inferred input shape from model.fit input np.zeros((64, 1))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Dense layer with 1 unit, matching the example
        self.dense = tf.keras.layers.Dense(units=1)

    def call(self, inputs, training=False):
        return self.dense(inputs)

def my_model_function():
    # Instantiate MyModel and compile similarly to the example
    model = MyModel()
    model.compile(optimizer=tf.optimizers.SGD(learning_rate=1.0), loss=tf.losses.MeanSquaredError())
    return model

def GetInput():
    # Return random float32 tensor with shape (64, 1) matching the example batch size and input shape
    # Using uniform distribution [0,1) as a reasonable generic input
    return tf.random.uniform((64, 1), dtype=tf.float32)

