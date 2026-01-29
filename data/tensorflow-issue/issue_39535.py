# tf.random.normal(shape=[5, 3]) ← Inferred input shape from example usage

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple sequential-like model with two Dense layers matching example shape
        self.dense1 = tf.keras.layers.Dense(3, input_shape=(3,))
        self.dense2 = tf.keras.layers.Dense(3)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def MMSE(preds, targets, mask_value=0.0):
    # Mask entries where targets equal mask_value (default 0.0)
    mask = tf.cast(tf.not_equal(targets, mask_value), tf.float32)
    num_rating = tf.reduce_sum(mask)
    # Mean squared error on masked positions, safe divide in case num_rating=0
    loss = tf.reduce_sum(tf.square(mask * (preds - targets))) / tf.maximum(num_rating, 1.0)
    return loss

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile with Adam optimizer and MMSE loss as per original example
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=MMSE)
    return model

def GetInput():
    # Return a random tensor input that matches MyModel input shape (batch=5, features=3)
    # Using tf.random.normal as in original example, rounded per their code, but rounding is optional
    data = tf.math.round(tf.random.normal(shape=[5, 3]))
    return data

