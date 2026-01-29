# tf.random.uniform((B, 7), dtype=tf.float32) and tf.random.uniform((B, 17), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define two input branches corresponding to inputs of shapes (7,) and (17,)
        # The order and naming of inputs can cause issues when converting to TFLite,
        # so we implement a model that explicitly uses signatures (tf.function with input signatures) to preserve input order.

        # Dense layer applied after concatenation
        self.dense = tf.keras.layers.Dense(27, name="output")

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 7], dtype=tf.float32, name="input_word_ids"),
        tf.TensorSpec(shape=[None, 17], dtype=tf.float32, name="input_bboxes"),
    ])
    def call(self, input_word_ids, input_bboxes):
        # Concatenate inputs on last axis and apply dense layer
        x = tf.concat([input_word_ids, input_bboxes], axis=-1)
        return self.dense(x)

def my_model_function():
    # Return an instance of MyModel with a signature-aware call method
    model = MyModel()
    # Build model by calling once with sample input shapes
    _ = model(tf.zeros([1,7]), tf.zeros([1,17]))
    # Set concrete function as a callable attribute for TFLite signature preservation
    model.call.get_concrete_function(
        tf.TensorSpec(shape=[None,7], dtype=tf.float32, name="input_word_ids"),
        tf.TensorSpec(shape=[None,17], dtype=tf.float32, name="input_bboxes")
    )
    return model

def GetInput():
    # Return a tuple of inputs corresponding to the signature:
    # input_word_ids -> shape (B,7), input_bboxes -> shape (B,17)
    # Using batch size = 1 for this example
    batch_size = 1
    input_word_ids = tf.random.uniform(shape=(batch_size, 7), dtype=tf.float32)
    input_bboxes = tf.random.uniform(shape=(batch_size, 17), dtype=tf.float32)
    return input_word_ids, input_bboxes

