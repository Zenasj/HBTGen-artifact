# tf.random.normal((B, 512), dtype=tf.float32) ← The inputs are two tensors of shape [batch_size, 512]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense = tf.keras.layers.Dense(100)

    @tf.function(
        input_signature=[
            (tf.TensorSpec([None, 512], tf.float32, name="responses"),
             tf.TensorSpec([None, 512], tf.float32, name="contexts"))
        ]
    )
    def call(self, inputs):
        # inputs is a tuple of (responses, contexts)
        responses, contexts = inputs
        return self.dense(responses + contexts)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Returns a tuple matching the input signature of MyModel.call: 
    # Two tensors shaped [batch, 512] with float32 dtype
    batch_size = 1  # arbitrary batch size
    responses = tf.random.normal((batch_size, 512), dtype=tf.float32)
    contexts = tf.random.normal((batch_size, 512), dtype=tf.float32)
    return (responses, contexts)

