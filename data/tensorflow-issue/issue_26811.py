# tf.random.uniform((B, 10), dtype=tf.float32) ← input shape is (batch_size, 10)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, units=10):
        super(MyModel, self).__init__()
        self.units = units
        # Important fix from the issue:
        # add_weight() calls need explicit 'name' argument to avoid weight saving errors
        self.w = self.add_weight(name='w',
                                 shape=(None, self.units),  # shape adjusted in build
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='b',
                                 shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def build(self, input_shape):
        # Weights have to be created with known shape based on input_shape
        # Re-create weights with correct shapes to override placeholders created at __init__
        # This is a workaround since add_weight calls in __init__ did not have shapes
        # (Better practice: define add_weight only inside build - but to stay closer to original)
        self.w = self.add_weight(name='w',
                                 shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='b',
                                 shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
        super(MyModel, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        # Linear layer: return inputs * w + b
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(MyModel, self).get_config()
        config.update({'units': self.units})
        return config

def my_model_function():
    # Return an instance of MyModel with default units=10
    return MyModel(units=10)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Based on issue, input shape is (batch_size, 10), dtype float32
    # Choose batch size = 32 arbitrarily for common batch processing
    return tf.random.uniform((32, 10), dtype=tf.float32)

