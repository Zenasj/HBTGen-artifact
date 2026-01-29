# tf.random.uniform((B, 1), dtype=tf.float32) ← Input shape is (batch_size, 1), matching x with shape (100, 1) for regression

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        activ = 'tanh'
        # From the original reg_model: input shape=(x.shape) means (None,1), here we hardcode (1,)
        # 7 hidden layers with 100 units and tanh activation, 1 output neuron without activation
        self.dense1 = tf.keras.layers.Dense(100, activation=activ, input_shape=(1,))
        self.dense2 = tf.keras.layers.Dense(100, activation=activ)
        self.dense3 = tf.keras.layers.Dense(100, activation=activ)
        self.dense4 = tf.keras.layers.Dense(100, activation=activ)
        self.dense5 = tf.keras.layers.Dense(100, activation=activ)
        self.dense6 = tf.keras.layers.Dense(100, activation=activ)
        self.dense7 = tf.keras.layers.Dense(100, activation=activ)
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        x = self.dense7(x)
        return self.out(x)

def my_model_function():
    # Instantiate the model and compile it to match original training setup
    model = MyModel()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanAbsoluteError())
    return model

def GetInput():
    # Generate a random input tensor matching the model input shape (batch_size, 1)
    # Batch size is assumed to be 1 for representative dataset usage, but can be arbitrary
    return tf.random.uniform((1, 1), dtype=tf.float32)

