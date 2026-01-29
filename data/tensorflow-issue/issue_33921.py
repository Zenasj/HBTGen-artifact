# tf.random.uniform((64, 3), dtype=tf.float32) ← inferred input shape and dtype from test case

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Recreate a small MLP similar to testing_utils.get_small_sequential_mlp(num_hidden=10, num_classes=2, input_dim=3)
        self.dense1 = tf.keras.layers.Dense(10, activation='relu')
        self.dense2 = tf.keras.layers.Dense(2, activation='linear')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        return self.dense2(x)


def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile model with sgd optimizer and mse loss like in the test case
    model.compile(optimizer='sgd', loss='mse')
    return model


def GetInput():
    # Return input tensor shaped (64, 3) matching test case input
    # dtype float32 by default, consistent with typical keras input
    return tf.random.uniform((64, 3), dtype=tf.float32)

