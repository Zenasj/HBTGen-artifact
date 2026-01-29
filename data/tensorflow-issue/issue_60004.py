# tf.random.uniform((10, 5), dtype=tf.float32) ← inferred input shape based on reproducible example inputs

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # The original issue involved trainability of Variables vs. weights.
        # Recommended approach: use add_weight for trainable parameters,
        # and create Variables with trainable=True if you want to change trainability.
        self.b = self.add_weight(shape=(1, 5), trainable=True, name="b")
        self.a = tf.Variable([1.0], trainable=True, name="a_var")
        # Note: after creation, tf.Variable.trainable attribute is readonly.
        # So to change trainability, recreate or use model/layer trainable flags instead.

    def call(self, inputs, training=None, mask=None):
        return self.a * self.b * inputs

def my_model_function():
    # Return instance of MyModel
    return MyModel()

def GetInput():
    # Return random input tensor with shape matching model input (10, 5) here.
    # Use dtype=tf.float32 to match default TensorFlow types.
    return tf.random.uniform((10, 5), dtype=tf.float32)

