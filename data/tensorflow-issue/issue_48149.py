# tf.random.uniform((B, 20), dtype=tf.float32) ← inferred input shape based on example: batch size unknown (B), 20 features

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Simple fully connected layer matching original example
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs):
        # Forward pass returns logits for 10 classes
        return self.dense(inputs)

    def predict_step(self, data):
        # Custom prediction logic: call model and take argmax of logits (class prediction)
        logits = self(data, training=False)
        return tf.argmax(logits, axis=-1)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 20], dtype=tf.float32)])
    def serve(self, inputs):
        # Custom serving function to preserve predict logic when saving/loading model
        logits = self(inputs, training=False)
        return tf.argmax(logits, axis=-1)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching the model input shape, compatible with tf.function signature
    # Using float32 as in original example, batch size arbitrarily set to 8 for testing
    return tf.random.uniform((8, 20), dtype=tf.float32)

